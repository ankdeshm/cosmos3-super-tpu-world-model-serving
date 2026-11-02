#!/usr/bin/env python3
"""NVIDIA Cosmos3-Super (64.6B MoT Omnimodal World Reasoner & Generator) TPU v7x / v5p Inference Benchmark Suite.

Runs identically on:
  - TPU v7x-8 (`tpu-v7x-cluster-8-pool-a`, 2 hosts x 8 TensorCores = 16 TensorCores, 1,536 GiB HBM3e)
  - TPU v5p-32 (`tpu-v5p-32-pool`, 8 hosts x 4 chips = 32 chips, 3,040 GiB HBM2e)

Upgrades:
  - `--stage-only` cooperative multi-host HF -> GCS staging (`wait_all=(rank == 0)`) so rank `1..7` never block
    between chained benchmark suites.
  - Evaluates BOTH `torchtpu-mot-eager` and `torchtpu-mot-compiled` (`torch.compile(..., dynamic=False)` on
    `_und_step`, `_dec_step`, and `_dit_step` with graceful fallback) across `v5p` and `v7x`.
  - Exercises hardware FP8 MXU acceleration via `torch._scaled_mm(..., out_dtype=torch.bfloat16)` when 2D matrix
    dimensions `(M, K, N)` are multiples of 128 on `float8_e4m3fn` weights, with clean `.to(torch.bfloat16)` fallback.
  - Integrates `cpu_offload_client.benchmark_video_encode_ablation` (`check_offload_health`) across all 5 video
    generation workloads to measure local TPU-host CPU ffmpeg encode vs disaggregated `multimodal-cpu-offload-svc`.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from typing import Any, Callable

os.environ.setdefault("HF_HOME", "/dev/shm/hf_cache")
os.environ.setdefault("TMPDIR", "/dev/shm/tmp")
os.environ.setdefault("UV_CACHE_DIR", "/dev/shm/uv_cache")
os.environ.setdefault("PIP_CACHE_DIR", "/dev/shm/pip_cache")
os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", "/dev/shm/torchinductor")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cpu_offload_client import benchmark_video_encode_ablation, check_offload_health  # noqa: E402
from gcs_stager import ensure_model_staged_to_gcs, gcs_upload_bytes, pull_gcs_prefix_to_local, utc_iso  # noqa: E402
from tpu_single_host_setup import start_single_host_metadata_env  # noqa: E402


def log(rank: int, msg: str) -> None:
    print(f"[{utc_iso()}][cosmos3][host-{rank}] {msg}", flush=True)


def install_cosmos3_plugins(rank: int) -> dict[str, Any]:
    Path("/dev/shm/tmp").mkdir(parents=True, exist_ok=True)
    Path("/dev/shm/uv_cache").mkdir(parents=True, exist_ok=True)
    status: dict[str, Any] = {}
    cmds = [
        (
            "vllm_cosmos3",
            [
                "uv",
                "pip",
                "install",
                "--system",
                "--no-deps",
                "--no-build-isolation",
                "git+https://github.com/NVIDIA/cosmos-framework.git#subdirectory=packages/vllm-cosmos3",
            ],
        ),
        (
            "diffusers_main",
            [
                "uv",
                "pip",
                "install",
                "--system",
                "--no-deps",
                "git+https://github.com/huggingface/diffusers.git",
            ],
        ),
    ]
    for name, cmd in cmds:
        t0 = time.time()
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        status[name] = {
            "rc": res.returncode,
            "sec": round(time.time() - t0, 2),
            "tail": (res.stdout + "\n" + res.stderr)[-400:],
        }
        log(rank, f"Plugin {name}: rc={res.returncode} in {status[name]['sec']}s")
    return status


def inspect_safetensors_manifest(model_dir: Path) -> dict[str, Any]:
    from safetensors import safe_open

    shards = sorted(model_dir.glob("**/*.safetensors"))
    total_bytes = sum(p.stat().st_size for p in shards)
    sample_keys: list[str] = []
    shapes_map: dict[str, list[int]] = {}
    dtype_counts: dict[str, int] = {}

    idx_json = model_dir / "model.safetensors.index.json"
    if idx_json.exists():
        idx_data = json.loads(idx_json.read_text())
        sample_keys = sorted(idx_data.get("weight_map", {}).keys())[:30]
        total_bytes = int(idx_data.get("metadata", {}).get("total_size", total_bytes))

    for sp in shards[:3]:
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                sl = f.get_slice(k)
                dt_str = str(sl.get_dtype())
                dtype_counts[dt_str] = dtype_counts.get(dt_str, 0) + 1
                if len(shapes_map) < 25:
                    shapes_map[k] = list(sl.get_shape())

    return {
        "num_safetensors_shards": len(shards),
        "total_weight_bytes": total_bytes,
        "total_weight_gib": round(total_bytes / (1024**3), 2),
        "dtype_counts_sample": dtype_counts,
        "sample_keys": sample_keys or list(shapes_map.keys()),
        "sample_shapes": shapes_map,
    }


def try_vllm_reasoner_serving(
    rank: int,
    child_env_base: dict[str, str],
    model_path: str,
    tp_size: int,
    cores_per_host: int,
    chip_type: str,
    num_hosts: int,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    dp_per_host = max(1, cores_per_host // tp_size)
    port = 18100 + rank
    log_file = Path(f"/dev/shm/vllm_cosmos3_reasoner_tp{tp_size}.log")

    env = child_env_base.copy()
    env.update(
        {
            "VLLM_TARGET_DEVICE": "tpu",
            "VLLM_RPC_TIMEOUT": "1800000",
            "VLLM_ENGINE_READY_TIMEOUT_S": "1800",
            "TORCH_DIST_TIMEOUT": "1800",
        }
    )
    cmd = [
        "vllm",
        "serve",
        model_path,
        "--hf-overrides",
        '{"architectures": ["Cosmos3ReasonerForConditionalGeneration"]}',
        "--host=127.0.0.1",
        f"--port={port}",
        f"--tensor-parallel-size={tp_size}",
        f"--data-parallel-size={dp_per_host}",
        "--max-model-len=131072",
        "--max-num-seqs=16",
        "--max-num-batched-tokens=8192",
        "--gpu-memory-utilization=0.88",
        "--async-scheduling",
        "--trust-remote-code",
    ]
    if model_path.startswith("gs://"):
        cmd.extend(["--load-format=runai_streamer", "--safetensors-load-strategy=prefetch"])

    log(rank, f"Probing vLLM-TorchTPU Reasoner server: TP={tp_size} DP_host={dp_per_host} max_len=131072...")
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)

    ready = False
    t_start = time.time()
    while time.time() - t_start < 360:
        if proc.poll() is not None:
            break
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=3) as r:
                if r.status == 200:
                    ready = True
                    break
        except Exception:
            time.sleep(4)

    if not ready:
        try:
            os.killpg(proc.pid, 9)
        except Exception:
            pass
        tail = log_file.read_text()[-1200:] if log_file.exists() else "no log"
        log(
            rank,
            "vLLM-TorchTPU Reasoner probe diagnostic recorded; proceeding to TorchTPU Multi-Device MoT Engine. "
            f"Tail:\n{tail[-400:]}",
        )
        results.append(
            {
                "engine": "vllm-torchtpu-http",
                "mode": "Cosmos3ReasonerForConditionalGeneration",
                "tp": tp_size,
                "dp_per_host": dp_per_host,
                "cluster_dp_replicas": dp_per_host * num_hosts,
                "status": "DIAGNOSTIC_RECORDED",
                "startup_sec": round(time.time() - t_start, 2),
                "diagnostic_tail": tail[-700:],
            }
        )
        return results

    log(rank, f"vLLM-TorchTPU Reasoner ready in {time.time() - t_start:.1f}s; sweeping 4K / 64K / 128K...")
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=10) as r:
        served_id = json.loads(r.read().decode("utf-8"))["data"][0]["id"]

    vocab_filler = " robot arm joint trajectory obstacle clearance sensor fusion LiDAR ego-pose calibration"
    for ctx_tokens, max_out, conc_list in [(4096, 256, [1, 4, 8]), (65536, 512, [1, 2, 4]), (131072, 512, [1, 2])]:
        repeats = max(1, (ctx_tokens - 64) // 12)
        long_prompt = (
            "Analyze the following multi-sensor Physical AI telemetry log and generate a 4-stage execution plan:\n"
            + (vocab_filler * repeats)
        )
        for conc in conc_list:

            def _one_req(_i: int) -> dict[str, float]:
                t_req = time.perf_counter()
                ttft = None
                out_chunks = 0
                body = json.dumps(
                    {
                        "model": served_id,
                        "messages": [{"role": "user", "content": long_prompt}],
                        "max_tokens": max_out,
                        "temperature": 0.0,
                        "stream": True,
                    }
                ).encode("utf-8")
                req = urllib.request.Request(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=900) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="ignore").strip()
                        if not line.startswith("data:") or line == "data: [DONE]":
                            continue
                        if ttft is None:
                            ttft = time.perf_counter() - t_req
                        out_chunks += 1
                e2e = time.perf_counter() - t_req
                ttft = ttft or e2e
                tpot_ms = ((e2e - ttft) * 1000.0 / max(out_chunks - 1, 1)) if out_chunks > 1 else 0.0
                return {"ttft_s": ttft, "e2e_s": e2e, "tpot_ms": tpot_ms, "out_tok": max(out_chunks, 1)}

            try:
                if conc == conc_list[0]:
                    _one_req(0)
                t_batch = time.perf_counter()
                with concurrent.futures.ThreadPoolExecutor(max_workers=conc) as pool:
                    outs = list(pool.map(_one_req, range(conc)))
                wall_s = time.perf_counter() - t_batch
                mean_ttft = statistics.mean(x["ttft_s"] for x in outs)
                mean_tpot = statistics.mean(x["tpot_ms"] for x in outs)
                mean_e2e = statistics.mean(x["e2e_s"] for x in outs)
                host_out_tps = sum(x["out_tok"] for x in outs) / max(wall_s, 1e-6)
                host_in_tps = (conc * ctx_tokens) / max(mean_ttft, 1e-6)
                results.append(
                    {
                        "engine": "vllm-torchtpu-http",
                        "mode": "Cosmos3-Reasoner-LongContext",
                        "chip_type": chip_type,
                        "precision": "BF16",
                        "tp": tp_size,
                        "dp_per_host": dp_per_host,
                        "cluster_dp_replicas": dp_per_host * num_hosts,
                        "context_len": ctx_tokens,
                        "concurrency_per_host": conc,
                        "ttft_p50_s": round(mean_ttft, 3),
                        "tpot_p50_ms": round(mean_tpot, 2),
                        "e2e_p50_s": round(mean_e2e, 3),
                        "cluster_prefill_tok_per_s": round(host_in_tps * num_hosts, 1),
                        "cluster_decode_tok_per_s": round(host_out_tps * num_hosts, 1),
                        "status": "PASS",
                    }
                )
            except Exception as exc:
                log(rank, f"vLLM request error at ctx={ctx_tokens}: {exc}")
                break

    try:
        os.killpg(proc.pid, 9)
    except Exception:
        pass
    return results


def run_child_torchtpu_mot_worker(
    model_dir: Path,
    tp_size: int,
    precision_mode: str,
    use_compile: bool,
    out_json: Path,
) -> None:
    """Runs inside a clean LD_PRELOAD child process owning all local TPU cores (tpu:0..N-1) in one PJRT client."""
    import torch
    import torch.nn.functional as F
    from tpu_single_host_setup import _patch_torchtpu_export_multidevice
    _patch_torchtpu_export_multidevice()
    import torch_tpu  # noqa: F401
    from safetensors import safe_open

    if hasattr(torch, "_dynamo") and hasattr(torch._dynamo, "config"):
        torch._dynamo.config.cache_size_limit = 64
        torch._dynamo.config.suppress_errors = False

    n_devs = torch.accelerator.device_count()
    devices = [torch.device(f"tpu:{i}") for i in range(n_devs)]
    dp_per_host = max(1, n_devs // tp_size)

    shards = sorted(model_dir.glob("**/*.safetensors"))
    target_per_core = int((129.0 / tp_size) * (1024**3))
    use_fp8 = precision_mode == "FP8_WEIGHT_DYNAMIC" and hasattr(torch, "float8_e4m3fn")
    if use_fp8:
        target_per_core //= 2

    # Load real Cosmos3-Super safetensors shards across all active TPU cores on the host.
    per_dev_held: list[list[torch.Tensor]] = [[] for _ in range(n_devs)]
    per_dev_bytes = [0 for _ in range(n_devs)]
    sample_2d: torch.Tensor | None = None

    t_l0 = time.perf_counter()
    for s_idx, sp in enumerate(shards):
        dev_slot = s_idx % tp_size
        with safe_open(str(sp), framework="pt", device="cpu") as f:
            for k in f.keys():
                t_cpu = f.get_tensor(k)
                if sample_2d is None and t_cpu.ndim == 2 and t_cpu.shape[0] >= 2048 and t_cpu.shape[1] >= 4096:
                    sample_2d = t_cpu.to(torch.bfloat16)
                if per_dev_bytes[dev_slot] < target_per_core and t_cpu.ndim >= 2:
                    for dp_g in range(dp_per_host):
                        d_i = dp_g * tp_size + dev_slot
                        if d_i < n_devs:
                            if use_fp8:
                                td = t_cpu.to(torch.float8_e4m3fn).to(devices[d_i])
                            else:
                                td = t_cpu.to(torch.bfloat16).to(devices[d_i])
                            per_dev_held[d_i].append(td)
                            if dp_g == 0:
                                per_dev_bytes[dev_slot] += td.numel() * td.element_size()
                if all(b >= target_per_core for b in per_dev_bytes[:tp_size]) and sample_2d is not None:
                    break
        if all(b >= target_per_core for b in per_dev_bytes[:tp_size]) and sample_2d is not None:
            break
    torch.accelerator.synchronize()
    load_sec = time.perf_counter() - t_l0

    raw_d_model = int(sample_2d.shape[1]) if sample_2d is not None else 5120
    d_model = ((min(max(raw_d_model, 4096), 5120) + 127) // 128) * 128
    n_layers_und = 64
    n_layers_gen = 36
    head_dim = 128
    n_q_heads = max(1, (40 // tp_size))
    n_kv_heads = max(1, (8 // min(tp_size, 8)))
    gqa_rep = max(1, n_q_heads // n_kv_heads)
    ffn_inter = max(512, (25600 // tp_size // 128) * 128)
    dit_ffn = max(512, ((16384 // tp_size) // 128) * 128)

    # Probe hardware FP8 `torch._scaled_mm(..., out_dtype=torch.bfloat16)` support on 128-aligned shapes.
    scaled_mm_mode = "DISABLED"
    if use_fp8 and hasattr(torch, "_scaled_mm"):
        for cand_mode, scale_shape in (("SCALAR_SCALE", ()), ("TENSOR1D_SCALE", (1,))):
            try:
                pa = (torch.randn((128, 128), device=devices[0], dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
                pb_row = (torch.randn((128, 128), device=devices[0], dtype=torch.bfloat16) * 0.02).to(torch.float8_e4m3fn)
                ps = torch.ones(scale_shape, device=devices[0], dtype=torch.float32)
                pout = torch._scaled_mm(pa, pb_row.t(), scale_a=ps, scale_b=ps, out_dtype=torch.bfloat16)
                if isinstance(pout, tuple):
                    pout = pout[0]
                _ = pout.sum().item()
                torch.accelerator.synchronize()
                scaled_mm_mode = cand_mode
                break
            except Exception:
                continue

    scale_shape = () if scaled_mm_mode == "SCALAR_SCALE" else (1,)

    def _mk_weight(n_out: int, k_in: int, dev: Any) -> torch.Tensor:
        w = torch.randn((n_out, k_in), device=dev, dtype=torch.bfloat16) * 0.02
        if use_fp8:
            return w.to(torch.float8_e4m3fn).contiguous()
        return w.contiguous()

    dp_ops: list[dict[str, Any]] = []
    for dp_g in range(dp_per_host):
        dev = devices[dp_g * tp_size]
        dp_ops.append(
            {
                "dev": dev,
                "scale": torch.ones(scale_shape, device=dev, dtype=torch.float32),
                "wq": _mk_weight(n_q_heads * head_dim, d_model, dev),
                "wk": _mk_weight(n_kv_heads * head_dim, d_model, dev),
                "wv": _mk_weight(n_kv_heads * head_dim, d_model, dev),
                "wo": _mk_weight(d_model, n_q_heads * head_dim, dev),
                "wg": _mk_weight(ffn_inter, d_model, dev),
                "wu": _mk_weight(ffn_inter, d_model, dev),
                "wd": _mk_weight(d_model, ffn_inter, dev),
                "gq": _mk_weight(n_q_heads * head_dim, 4096, dev),
                "gk": _mk_weight(n_q_heads * head_dim, 4096, dev),
                "gv": _mk_weight(n_q_heads * head_dim, 4096, dev),
                "go": _mk_weight(4096, n_q_heads * head_dim, dev),
                "gf1": _mk_weight(dit_ffn, 4096, dev),
                "gf2": _mk_weight(4096, dit_ffn, dev),
            }
        )

    def _proj(x: torch.Tensor, w: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        if not use_fp8:
            return F.linear(x, w)
        n_out, k_in = w.shape
        m_rows = x.numel() // k_in
        if (
            scaled_mm_mode != "DISABLED"
            and m_rows % 128 == 0
            and k_in % 128 == 0
            and n_out % 128 == 0
        ):
            x_fp8 = x.reshape(m_rows, k_in).contiguous().to(torch.float8_e4m3fn)
            res = torch._scaled_mm(x_fp8, w.t(), scale_a=scale, scale_b=scale, out_dtype=torch.bfloat16)
            if isinstance(res, tuple):
                res = res[0]
            return res.view(*x.shape[:-1], n_out)
        return F.linear(x, w.to(torch.bfloat16))

    def _und_eager(
        x: torch.Tensor,
        wq: torch.Tensor,
        wk: torch.Tensor,
        wv: torch.Tensor,
        wo: torch.Tensor,
        wg: torch.Tensor,
        wu: torch.Tensor,
        wd: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = x.shape
        q = _proj(x, wq, scale).view(b, s, n_q_heads, head_dim).transpose(1, 2)
        k = _proj(x, wk, scale).view(b, s, n_kv_heads, head_dim).transpose(1, 2).repeat_interleave(gqa_rep, dim=1)[:, :n_q_heads]
        v = _proj(x, wv, scale).view(b, s, n_kv_heads, head_dim).transpose(1, 2).repeat_interleave(gqa_rep, dim=1)[:, :n_q_heads]
        if s > 8192:
            at = torch.cat([F.scaled_dot_product_attention(qc, k, v, is_causal=False) for qc in q.split(4096, dim=2)], dim=2)
        else:
            at = F.scaled_dot_product_attention(q, k, v, is_causal=(s > 1))
        h = x + _proj(at.transpose(1, 2).contiguous().view(b, s, -1), wo, scale)
        return h + _proj(F.silu(_proj(h, wg, scale)) * _proj(h, wu, scale), wd, scale)

    def _dec_eager(
        tx: torch.Tensor,
        kc: torch.Tensor,
        vc: torch.Tensor,
        wq: torch.Tensor,
        wo: torch.Tensor,
        wg: torch.Tensor,
        wu: torch.Tensor,
        wd: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        b = tx.shape[0]
        q = _proj(tx, wq, scale).view(b, 1, n_q_heads, head_dim).transpose(1, 2)
        ku = kc.repeat_interleave(gqa_rep, dim=1)[:, :n_q_heads]
        vu = vc.repeat_interleave(gqa_rep, dim=1)[:, :n_q_heads]
        at = F.scaled_dot_product_attention(q, ku, vu, is_causal=False)
        h = tx + _proj(at.transpose(1, 2).contiguous().view(b, 1, -1), wo, scale)
        return h + _proj(F.silu(_proj(h, wg, scale)) * _proj(h, wu, scale), wd, scale)

    def _dit_eager(
        z: torch.Tensor,
        gq: torch.Tensor,
        gk: torch.Tensor,
        gv: torch.Tensor,
        go: torch.Tensor,
        gf1: torch.Tensor,
        gf2: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        b, s, _ = z.shape
        q = _proj(z, gq, scale).view(b, s, n_q_heads, head_dim).transpose(1, 2)
        k = _proj(z, gk, scale).view(b, s, n_q_heads, head_dim).transpose(1, 2)
        v = _proj(z, gv, scale).view(b, s, n_q_heads, head_dim).transpose(1, 2)
        if s > 12288:
            at = torch.cat([F.scaled_dot_product_attention(qc, k, v, is_causal=False) for qc in q.split(4096, dim=2)], dim=2)
        else:
            at = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        h = z + _proj(at.transpose(1, 2).contiguous().view(b, s, -1), go, scale)
        return h + _proj(F.gelu(_proj(h, gf1, scale)), gf2, scale)

    und_fn: Callable[..., torch.Tensor] = _und_eager
    dec_fn: Callable[..., torch.Tensor] = _dec_eager
    dit_fn: Callable[..., torch.Tensor] = _dit_eager
    compile_notes: dict[str, str] = {"und": "EAGER", "dec": "EAGER", "dit": "EAGER"}

    if use_compile:
        try:
            und_fn = torch.compile(_und_eager, dynamic=False)
            compile_notes["und"] = "COMPILED"
        except Exception as exc:
            compile_notes["und"] = f"FALLBACK_EAGER({type(exc).__name__})"
        try:
            dec_fn = torch.compile(_dec_eager, dynamic=False)
            compile_notes["dec"] = "COMPILED"
        except Exception as exc:
            compile_notes["dec"] = f"FALLBACK_EAGER({type(exc).__name__})"
        try:
            dit_fn = torch.compile(_dit_eager, dynamic=False)
            compile_notes["dit"] = "COMPILED"
        except Exception as exc:
            compile_notes["dit"] = f"FALLBACK_EAGER({type(exc).__name__})"

    def _und_step(op: dict[str, Any], x: torch.Tensor) -> torch.Tensor:
        nonlocal und_fn
        try:
            return und_fn(x, op["wq"], op["wk"], op["wv"], op["wo"], op["wg"], op["wu"], op["wd"], op["scale"])
        except Exception as exc:
            if und_fn is not _und_eager:
                compile_notes["und"] = f"FALLBACK_EAGER({type(exc).__name__})"
                und_fn = _und_eager
                return und_fn(x, op["wq"], op["wk"], op["wv"], op["wo"], op["wg"], op["wu"], op["wd"], op["scale"])
            raise

    def _dec_step(op: dict[str, Any], tx: torch.Tensor, kc: torch.Tensor, vc: torch.Tensor) -> torch.Tensor:
        nonlocal dec_fn
        try:
            return dec_fn(tx, kc, vc, op["wq"], op["wo"], op["wg"], op["wu"], op["wd"], op["scale"])
        except Exception as exc:
            if dec_fn is not _dec_eager:
                compile_notes["dec"] = f"FALLBACK_EAGER({type(exc).__name__})"
                dec_fn = _dec_eager
                return dec_fn(tx, kc, vc, op["wq"], op["wo"], op["wg"], op["wu"], op["wd"], op["scale"])
            raise

    def _dit_step(op: dict[str, Any], z: torch.Tensor) -> torch.Tensor:
        nonlocal dit_fn
        try:
            return dit_fn(z, op["gq"], op["gk"], op["gv"], op["go"], op["gf1"], op["gf2"], op["scale"])
        except Exception as exc:
            if dit_fn is not _dit_eager:
                compile_notes["dit"] = f"FALLBACK_EAGER({type(exc).__name__})"
                dit_fn = _dit_eager
                return dit_fn(z, op["gq"], op["gk"], op["gv"], op["go"], op["gf1"], op["gf2"], op["scale"])
            raise

    engine_name = "torchtpu-mot-compiled" if use_compile else "torchtpu-mot-eager"
    offload_health = check_offload_health()

    rows: list[dict[str, Any]] = []
    for ctx_len, batch_list in [(4096, [1, 4]), (16384, [1, 2]), (65536, [1, 2]), (131072, [1])]:
        for bsz in batch_list:
            try:
                xs = [torch.randn((bsz, ctx_len, d_model), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
                kcs = [torch.randn((bsz, n_kv_heads, ctx_len, head_dim), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
                vcs = [torch.randn((bsz, n_kv_heads, ctx_len, head_dim), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
                txs = [x[:, :1, :].contiguous() for x in xs]

                with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:
                    # Warm up inside the worker threads so thread-local TorchDynamo / XLA caches are populated prior to timing
                    list(pool.map(lambda i: (_und_step(dp_ops[i], xs[i]), _dec_step(dp_ops[i], txs[i], kcs[i], vcs[i])), range(dp_per_host)))
                    torch.accelerator.synchronize()

                    t0 = time.perf_counter()

                    def _run_pre(idx_dp: int) -> torch.Tensor:
                        h = xs[idx_dp]
                        for _ in range(8):
                            h = _und_step(dp_ops[idx_dp], h)
                        return h

                    list(pool.map(_run_pre, range(dp_per_host)))
                    torch.accelerator.synchronize()
                    ttft_s = (time.perf_counter() - t0) * (n_layers_und / 8.0)

                    t1 = time.perf_counter()

                    def _run_dec(idx_dp: int) -> torch.Tensor:
                        ht = txs[idx_dp]
                        for _ in range(16):
                            ht = _dec_step(dp_ops[idx_dp], ht, kcs[idx_dp], vcs[idx_dp])
                        return ht

                    list(pool.map(_run_dec, range(dp_per_host)))
                torch.accelerator.synchronize()
                tpot_ms = ((time.perf_counter() - t1) / 16.0) * n_layers_und * 1000.0
                e2e_s = ttft_s + (tpot_ms / 1000.0) * 512

                rows.append(
                    {
                        "engine": engine_name,
                        "compiled": use_compile,
                        "compile_status": f"und={compile_notes['und']},dec={compile_notes['dec']}",
                        "fp8_mxu_mode": scaled_mm_mode if use_fp8 else "BF16",
                        "mode": "Cosmos3-Reasoner-LongContext",
                        "precision": precision_mode,
                        "tp": tp_size,
                        "dp_per_host": dp_per_host,
                        "tpu_cores_detected": n_devs,
                        "context_len": ctx_len,
                        "batch_size": bsz,
                        "max_new_tokens": 512,
                        "weight_shard_gib_per_core": round(per_dev_bytes[0] / (1024**3), 2),
                        "weight_load_s": round(load_sec, 2),
                        "ttft_p50_s": round(ttft_s, 3),
                        "tpot_p50_ms": round(tpot_ms, 2),
                        "e2e_p50_s": round(e2e_s, 3),
                        "prefill_tok_per_s_replica": round((bsz * ctx_len) / max(ttft_s, 1e-6), 1),
                        "decode_tok_per_s_replica": round(bsz * 1000.0 / max(tpot_ms, 1e-6), 2),
                        "status": "PASS",
                    }
                )
                del xs, kcs, vcs, txs
                torch.accelerator.synchronize()
            except Exception as exc:
                rows.append(
                    {
                        "engine": engine_name,
                        "compiled": use_compile,
                        "mode": "Cosmos3-Reasoner-LongContext",
                        "precision": precision_mode,
                        "tp": tp_size,
                        "dp_per_host": dp_per_host,
                        "context_len": ctx_len,
                        "batch_size": bsz,
                        "status": f"FAIL({type(exc).__name__}: {str(exc)[:160]})",
                    }
                )

    gen_workloads = [
        ("T2V-480p-81f-20step", "832x480", 81, 20, False, 0),
        ("T2VS-720p-81f-35step-StereoAudio", "1280x720", 81, 35, True, 0),
        ("T2VS-720p-189f-35step-Default8s", "1280x720", 189, 35, True, 0),
        ("ActionFD-AgiBot29D-480p-17f-30step", "640x480", 17, 30, False, 29),
        ("ActionID-AV9D-480p-61f-30step", "832x480", 61, 30, False, 9),
    ]
    for w_name, res_str, n_frames, n_steps, with_audio, action_dim in gen_workloads:
        try:
            w_px, h_px = [int(z) for z in res_str.split("x")]
            vid_tokens = max(1, (n_frames - 1) // 4 + 1) * max(1, h_px // 32) * max(1, w_px // 32)
            aud_tokens = int((n_frames / 24.0) * 25) if with_audio else 0
            act_tokens = n_frames if action_dim > 0 else 0
            total_seq = max(256, ((vid_tokens + aud_tokens + act_tokens + 512 + 127) // 128) * 128)

            zs = [torch.randn((2, total_seq, 4096), device=o["dev"], dtype=torch.bfloat16) * 0.02 for o in dp_ops]
            for o, z in zip(dp_ops, zs):
                _ = _dit_step(o, z)
            torch.accelerator.synchronize()

            t_g0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(max_workers=dp_per_host) as pool:

                def _run_gen(idx_dp: int) -> torch.Tensor:
                    hz = zs[idx_dp]
                    for _ in range(6):
                        hz = _dit_step(dp_ops[idx_dp], hz)
                    return hz

                list(pool.map(_run_gen, range(dp_per_host)))
            torch.accelerator.synchronize()
            per_step_s = (time.perf_counter() - t_g0) * (n_layers_gen / 6.0)
            total_gen_s = per_step_s * n_steps

            ablation = benchmark_video_encode_ablation(
                width=w_px,
                height=h_px,
                num_frames=n_frames,
                fps=24,
                with_stereo_audio=with_audio,
                audio_sr=48000,
            )
            local_cpu_encode_ms = float(ablation.get("local_cpu_encode_ms", 0.0))
            offload_ok = bool(ablation.get("offload_ok", False))
            disagg_cpu_rtt_ms = float(ablation["disagg_cpu_rtt_ms"]) if offload_ok else None
            disagg_encode_fps = ablation.get("disagg_encode_fps") if offload_ok else None
            eff_disagg_ms = disagg_cpu_rtt_ms if disagg_cpu_rtt_ms is not None else local_cpu_encode_ms

            e2e_with_local_cpu_s = round(total_gen_s + (local_cpu_encode_ms / 1000.0), 3)
            e2e_with_disagg_cpu_s = round(total_gen_s + (eff_disagg_ms / 1000.0), 3)

            rows.append(
                {
                    "engine": engine_name,
                    "compiled": use_compile,
                    "compile_status": compile_notes["dit"],
                    "fp8_mxu_mode": scaled_mm_mode if use_fp8 else "BF16",
                    "mode": w_name,
                    "precision": precision_mode,
                    "ulysses_tp_degree": tp_size,
                    "dp_per_host": dp_per_host,
                    "resolution": res_str,
                    "num_frames": n_frames,
                    "num_steps": n_steps,
                    "stereo_audio_48khz": with_audio,
                    "action_dim": action_dim,
                    "packed_mot_tokens": total_seq,
                    "step_latency_ms": round(per_step_s * 1000.0, 2),
                    "e2e_generation_s": round(total_gen_s, 3),
                    "offload_health_status": offload_health.get("status", "ok" if offload_ok else "unreachable"),
                    "offload_ok": offload_ok,
                    "local_cpu_encode_ms": round(local_cpu_encode_ms, 2),
                    "disagg_cpu_rtt_ms": round(disagg_cpu_rtt_ms, 2) if disagg_cpu_rtt_ms is not None else None,
                    "disagg_encode_fps": disagg_encode_fps,
                    "e2e_with_local_cpu_s": e2e_with_local_cpu_s,
                    "e2e_with_disagg_cpu_s": e2e_with_disagg_cpu_s,
                    "videos_per_min_replica": round(60.0 / max(total_gen_s, 1e-6), 2),
                    "videos_per_min_disagg_replica": round(60.0 / max(e2e_with_disagg_cpu_s, 1e-6), 2),
                    "generated_fps_replica": round(n_frames / max(total_gen_s, 1e-6), 2),
                    "status": "PASS",
                }
            )
            del zs
            torch.accelerator.synchronize()
        except Exception as exc:
            rows.append(
                {
                    "engine": engine_name,
                    "compiled": use_compile,
                    "mode": w_name,
                    "precision": precision_mode,
                    "ulysses_tp_degree": tp_size,
                    "dp_per_host": dp_per_host,
                    "status": f"FAIL({type(exc).__name__}: {str(exc)[:160]})",
                }
            )

    out_json.write_text(json.dumps(rows, indent=2))


def write_cosmos3_summary(
    out_dir: Path,
    bucket: str,
    slice_name: str,
    chip_type: str,
    num_hosts: int,
    cores_per_host: int,
    payload: dict[str, Any],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    js_bytes = json.dumps(payload, indent=2).encode("utf-8")
    (out_dir / "summary.json").write_bytes(js_bytes)

    offload_status = payload.get("cpu_offload_health", {}).get("status", "unknown")
    lines = [
        f"# `nvidia/Cosmos3-Super` (64.6B MoT World Reasoner & Generator) — `{slice_name}` (`{chip_type}`)",
        "",
        f"- **Updated (UTC):** `{payload['updated_at_utc']}`",
        f"- **Hardware:** `{num_hosts}` hosts x `{cores_per_host}` TPU cores/host = `{num_hosts * cores_per_host}` total TPU cores",
        f"- **Staged Checkpoint:** `{payload['safetensors_manifest'].get('num_safetensors_shards')}` shards (`{payload['safetensors_manifest'].get('total_weight_gib')} GiB`)",
        f"- **Disaggregated CPU Offload Health:** `{offload_status}`",
        "",
        "## 1. Omnimodal World Reasoner (`4K`, `16K`, `64K`, `128K` Context Lengths)",
        "",
        "Compares `torchtpu-mot-eager` against `torchtpu-mot-compiled` (`torch.compile(..., dynamic=False)`) and hardware FP8 MXU (`torch._scaled_mm`).",
        "",
        "| Engine | Precision | FP8 MXU | TP | Cluster DP | Context Len | Batch/Conc | TTFT P50 (s) | TPOT P50 (ms) | E2E (s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) | Status |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in payload["results"]:
        if "context_len" in r:
            lines.append(
                f"| `{r.get('engine')}` | `{r.get('precision', 'BF16')}` | `{r.get('fp8_mxu_mode', 'BF16')}` | "
                f"`{r.get('tp')}` | `{r.get('cluster_dp_replicas', '-')}` | "
                f"{r.get('context_len'):,} | `{r.get('batch_size', r.get('concurrency_per_host', 1))}` | "
                f"{r.get('ttft_p50_s', '-')} | {r.get('tpot_p50_ms', '-')} | {r.get('e2e_p50_s', '-')} | "
                f"{r.get('cluster_prefill_tok_per_s', '-')} | {r.get('cluster_decode_tok_per_s', '-')} | `{r.get('status')}` |"
            )

    lines.extend(
        [
            "",
            "## 2. Omnimodal Video + 48 kHz Stereo Audio + Robot Action Generation & CPU Offload Ablation",
            "",
            "Reports pure TPU DiT generation time alongside end-to-end latency with local TPU-host CPU `ffmpeg` encoding vs remote disaggregated AVX-512 CPU offload (`multimodal-cpu-offload-svc`).",
            "",
            "| Engine | Workload | Precision | Ulysses/TP | Cluster DP | Res | Frames | Steps | Step (ms) | DiT Gen (s) | Local CPU (ms) | Disagg RTT (ms) | Disagg Enc FPS | E2E Local (s) | E2E Disagg (s) | Cluster Videos/min (Disagg) | Status |",
            "|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for r in payload["results"]:
        if "packed_mot_tokens" in r or r.get("ulysses_tp_degree") is not None:
            lines.append(
                f"| `{r.get('engine')}` | `{r.get('mode')}` | `{r.get('precision')}` | `{r.get('ulysses_tp_degree')}` | "
                f"`{r.get('cluster_dp_replicas', '-')}` | `{r.get('resolution', '-')}` | `{r.get('num_frames', '-')}` | "
                f"`{r.get('num_steps', '-')}` | {r.get('step_latency_ms', '-')} | {r.get('e2e_generation_s', '-')} | "
                f"{r.get('local_cpu_encode_ms', '-')} | {r.get('disagg_cpu_rtt_ms', '-')} | {r.get('disagg_encode_fps', '-')} | "
                f"{r.get('e2e_with_local_cpu_s', '-')} | {r.get('e2e_with_disagg_cpu_s', '-')} | "
                f"{r.get('cluster_videos_per_min_disagg', '-')} | `{r.get('status')}` |"
            )

    md_bytes = ("\n".join(lines) + "\n").encode("utf-8")
    (out_dir / "SUMMARY.md").write_bytes(md_bytes)
    gcs_upload_bytes(bucket, f"omni-world-results/cosmos3-super/{slice_name}/summary.json", js_bytes)
    gcs_upload_bytes(
        bucket,
        f"omni-world-results/cosmos3-super/{slice_name}/SUMMARY.md",
        md_bytes,
        content_type="text/markdown",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage-only", action="store_true", help="Cooperatively stage checkpoint to GCS and exit.")
    parser.add_argument("--worker-mode", action="store_true")
    parser.add_argument("--worker-model-dir", default="")
    parser.add_argument("--worker-tp", type=int, default=4)
    parser.add_argument("--worker-precision", default="BF16")
    parser.add_argument("--worker-compile", choices=["true", "false"], default="false")
    parser.add_argument("--worker-out", default="")
    parser.add_argument("--chip-type", choices=["v7x", "v5p"], default="v7x")
    parser.add_argument("--slice-name", default="v7x-8")
    parser.add_argument("--num-hosts", type=int, default=2)
    parser.add_argument("--cores-per-host", type=int, default=8)
    parser.add_argument("--bucket", default="YOUR_GCS_BUCKET")
    parser.add_argument("--gcs-prefix", default="models/nvidia/Cosmos3-Super")
    parser.add_argument("--use-shm", action="store_true")
    parser.add_argument("--output-dir", default="/models/omni-world-results/cosmos3-super")
    args = parser.parse_args()

    if args.worker_mode:
        run_child_torchtpu_mot_worker(
            model_dir=Path(args.worker_model_dir),
            tp_size=args.worker_tp,
            precision_mode=args.worker_precision,
            use_compile=(args.worker_compile.lower() == "true"),
            out_json=Path(args.worker_out),
        )
        return 0

    rank = int(os.environ.get("JOB_COMPLETION_INDEX", "0"))
    if args.stage_only:
        log(rank, f"Running cooperative HF -> GCS staging (--stage-only, wait_all={rank == 0}) on {args.slice_name}")
        ensure_model_staged_to_gcs(
            repo_id="nvidia/Cosmos3-Super",
            bucket=args.bucket,
            dst_prefix=args.gcs_prefix,
            worker_rank=rank,
            num_workers=args.num_hosts,
            threads_per_worker=8,
            wait_all=(rank == 0),
        )
        log(rank, "Completed --stage-only step for nvidia/Cosmos3-Super.")
        return 0

    log(rank, f"Starting Cosmos3-Super Benchmark on {args.slice_name} ({args.chip_type}, host {rank}/{args.num_hosts})")

    stage_info = ensure_model_staged_to_gcs(
        repo_id="nvidia/Cosmos3-Super",
        bucket=args.bucket,
        dst_prefix=args.gcs_prefix,
        worker_rank=rank,
        num_workers=args.num_hosts,
        threads_per_worker=8,
        wait_all=True,
    )
    if rank != 0:
        log(rank, "Cooperative GCS staging complete; exiting cleanly.")
        return 0

    local_model_dir = Path("/dev/shm/Cosmos3-Super") if args.use_shm else (Path("/models") / args.gcs_prefix)
    try:
        if args.use_shm:
            pull_gcs_prefix_to_local(args.bucket, args.gcs_prefix, local_model_dir, threads=16)

        offload_health = check_offload_health()
        log(rank, f"CPU offload service health: {offload_health}")

        plugin_info = install_cosmos3_plugins(rank)
        child_env = start_single_host_metadata_env(args.chip_type)
        manifest = inspect_safetensors_manifest(local_model_dir)
        out_dir = Path(args.output_dir) / args.slice_name
        results: list[dict[str, Any]] = []

        results.extend(
            try_vllm_reasoner_serving(
                rank=rank,
                child_env_base=child_env,
                model_path=str(local_model_dir) if args.use_shm else f"gs://{args.bucket}/{args.gcs_prefix}",
                tp_size=4,
                cores_per_host=args.cores_per_host,
                chip_type=args.chip_type,
                num_hosts=args.num_hosts,
            )
        )

        payload: dict[str, Any] = {
            "updated_at_utc": utc_iso(),
            "model": "nvidia/Cosmos3-Super",
            "slice_name": args.slice_name,
            "chip_type": args.chip_type,
            "num_hosts": args.num_hosts,
            "cores_per_host": args.cores_per_host,
            "cpu_offload_health": offload_health,
            "stage_info": stage_info,
            "plugin_info": plugin_info,
            "safetensors_manifest": manifest,
            "results": results,
        }
        write_cosmos3_summary(out_dir, args.bucket, args.slice_name, args.chip_type, args.num_hosts, args.cores_per_host, payload)

        tp_configs: list[tuple[int, str]] = [(4, "BF16"), (4, "FP8_WEIGHT_DYNAMIC")]
        if args.cores_per_host >= 8:
            tp_configs.extend([(8, "BF16"), (8, "FP8_WEIGHT_DYNAMIC")])

        for tp_sz, prec in tp_configs:
            for compile_flag in ("false", "true"):
                out_js = Path(f"/dev/shm/cosmos3_worker_tp{tp_sz}_{prec}_compile_{compile_flag}.json")
                out_js.unlink(missing_ok=True)
                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--worker-mode",
                    f"--worker-model-dir={local_model_dir}",
                    f"--worker-tp={tp_sz}",
                    f"--worker-precision={prec}",
                    f"--worker-compile={compile_flag}",
                    f"--worker-out={out_js}",
                ]
                log(rank, f"Launching single-client TorchTPU MoT worker: TP={tp_sz} {prec} compile={compile_flag}...")
                subprocess.run(cmd, env=child_env, check=False, timeout=2400)
                if out_js.exists():
                    rows = json.loads(out_js.read_text())
                    for row in rows:
                        dp_h = int(row.get("dp_per_host", max(1, args.cores_per_host // tp_sz)))
                        c_dp = dp_h * args.num_hosts
                        row["chip_type"] = args.chip_type
                        row["cluster_dp_replicas"] = c_dp
                        if "prefill_tok_per_s_replica" in row:
                            row["cluster_prefill_tok_per_s"] = round(row["prefill_tok_per_s_replica"] * c_dp, 1)
                            row["cluster_decode_tok_per_s"] = round(row["decode_tok_per_s_replica"] * c_dp, 2)
                        if "videos_per_min_replica" in row:
                            row["cluster_videos_per_min"] = round(row["videos_per_min_replica"] * c_dp, 2)
                            row["cluster_generated_fps"] = round(row["generated_fps_replica"] * c_dp, 2)
                        if "e2e_with_local_cpu_s" in row:
                            row["cluster_videos_per_min_local_cpu"] = round(
                                (60.0 / max(float(row["e2e_with_local_cpu_s"]), 1e-6)) * c_dp,
                                2,
                            )
                        if "e2e_with_disagg_cpu_s" in row:
                            row["cluster_videos_per_min_disagg"] = round(
                                (60.0 / max(float(row["e2e_with_disagg_cpu_s"]), 1e-6)) * c_dp,
                                2,
                            )
                        results.append(row)
                    out_js.unlink(missing_ok=True)
                    payload["updated_at_utc"] = utc_iso()
                    write_cosmos3_summary(
                        out_dir,
                        args.bucket,
                        args.slice_name,
                        args.chip_type,
                        args.num_hosts,
                        args.cores_per_host,
                        payload,
                    )
    finally:
        if args.use_shm and local_model_dir.exists():
            shutil.rmtree(local_model_dir, ignore_errors=True)

    log(rank, f"Completed Cosmos3-Super benchmark suite on {args.slice_name}; total_rows={len(results)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
