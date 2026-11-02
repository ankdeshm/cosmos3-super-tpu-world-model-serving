# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x`

nvidia/Cosmos3-Super (64.6B Mixture-of-Transformers Omnimodal World Reasoner & Video/Audio/Action Generator) Serving on Cloud TPU v5p & v7x with Disaggregated CPU Offloading.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoint** | [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (`64.6B` BF16 / FP8) |
| **Backbone** | Dual-Tower Mixture-of-Transformers (MoT): World Reasoner (`4K / 16K / 64K / 128K` context) + Flow-Matching World Generator |
| **Output Modalities** | Synchronized World Video (`480p / 720p`) + `48 kHz` Stereo Audio + `29-DoF` Continuous Robot Action Trajectories |
| **TPU Execution** | `TorchTPU` (`eager` & `torch.compile(..., dynamic=False)`), `BF16` & `FP8_WEIGHT_DYNAMIC` (`out_dtype=bfloat16`), `TP=4` / `TP=8` |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512): `POST /encode` H.264 MP4 + `48 kHz` AAC muxing & `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

All non-TPU pre-processing and media encoding work is offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods (`scripts/multimodal_cpu_offload_server.py`), keeping TPU host CPUs and Python threads dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` batch image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera RGB image decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Benchmark Results (`v5p` & `v7x`)

> Live TPU `v5p` and `v7x` benchmark sweeps are currently executing on cluster slices (`v5p-32`, `v5p-16`, and `v7x-spot-4c`). Incremental checkpoints (`results/<slice>/summary.json` and `results/<slice>/SUMMARY.md`) are automatically synced here as each sweep stage finishes.

---

## 5. Repository Layout & Reproduction

- [`scripts/run_cosmos3_super_tpu_bench.py`](./scripts/run_cosmos3_super_tpu_bench.py) — Main TPU `v5p` / `v7x` benchmark driver (cooperative zero-disk HF-to-GCS staging, `/dev/shm` hydration, single-host PJRT isolation, `TorchTPU` / `vLLM` execution, and A/B CPU offload ablation).
- [`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py) — Disaggregated CPU offload HTTP server and client.
- [`scripts/gcs_stager.py`](./scripts/gcs_stager.py) & [`scripts/tpu_single_host_setup.py`](./scripts/tpu_single_host_setup.py) — Multi-host GCS shard stager and single-host `libtpu` metadata hook.
- [`k8s/`](./k8s/) — Kubernetes Deployment & Indexed Job manifests for CPU offload pool, TPU `v5p`, and TPU `v7x`.

```bash
# 1. Deploy the disaggregated CPU offload service
kubectl apply -f k8s/multimodal-cpu-offload-service.yaml

# 2. Launch the TPU v5p / v7x Indexed Job
kubectl apply -f k8s/v5p32-cosmos3-and-hunyuan-image3.yaml
```
