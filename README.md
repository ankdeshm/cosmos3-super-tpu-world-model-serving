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

Measured results linked 1:1 to raw JSON and Markdown summaries under [`results/`](./results/):

- **`v5p-32`** (`25` benchmark rows, updated `2026-11-02T19:06:01.483749+00:00`): [`results/v5p-32/SUMMARY.md`](./results/v5p-32/SUMMARY.md) · [`results/v5p-32/summary.json`](./results/v5p-32/summary.json)

# `nvidia/Cosmos3-Super` (64.6B MoT World Reasoner & Generator) — `v5p-32` (`v5p`)

- **Updated (UTC):** `2026-11-02T19:06:01.483749+00:00`
- **Hardware:** `8` hosts x `4` TPU cores/host = `32` total TPU cores
- **Staged Checkpoint:** `30` shards (`120.35 GiB`)
- **Disaggregated CPU Offload Health:** `ok`

## 1. Omnimodal World Reasoner (`4K`, `16K`, `64K`, `128K` Context Lengths)

Compares `torchtpu-mot-eager` against `torchtpu-mot-compiled` (`torch.compile(..., dynamic=False)`) and hardware FP8 MXU (`torch._scaled_mm`).

| Engine | Precision | FP8 MXU | TP | Cluster DP | Context Len | Batch/Conc | TTFT P50 (s) | TPOT P50 (ms) | E2E (s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 4,096 | `1` | 0.716 | 429.32 | 220.529 | 45796.8 | 18.64 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 4,096 | `4` | 2.709 | 121.07 | 64.695 | 48388.8 | 264.32 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 16,384 | `1` | 2.988 | 117.63 | 63.217 | 43860.0 | 68.0 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 16,384 | `2` | 5.585 | 101.17 | 57.387 | 46935.2 | 158.16 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 65,536 | `1` | 13.53 | 122.3 | 76.146 | 38750.4 | 65.44 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 65,536 | `2` | 26.813 | 123.84 | 90.218 | 39106.4 | 129.2 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 131,072 | `1` | 34.813 | 83.84 | 77.738 | 30120.0 | 95.44 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 4,096 | `1` | 0.353 | 717.86 | 367.898 | 92715.2 | 11.12 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 4,096 | `4` | 1.36 | 33.02 | 18.267 | 96406.4 | 969.04 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 16,384 | `1` | 1.631 | 711.27 | 365.802 | 80356.8 | 11.28 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 16,384 | `2` | 3.038 | 37.97 | 22.481 | 86276.8 | 421.36 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 65,536 | `1` | 8.774 | 742.23 | 388.798 | 59756.8 | 10.8 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 65,536 | `2` | 17.81 | 70.47 | 53.888 | 58876.0 | 227.04 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `4` | `8` | 131,072 | `1` | 25.647 | 1829.22 | 962.208 | 40884.8 | 4.4 | `PASS` |

## 2. Omnimodal Video + 48 kHz Stereo Audio + Robot Action Generation & CPU Offload Ablation

Reports pure TPU DiT generation time alongside end-to-end latency with local TPU-host CPU `ffmpeg` encoding vs remote disaggregated AVX-512 CPU offload (`multimodal-cpu-offload-svc`).

| Engine | Workload | Precision | Ulysses/TP | Cluster DP | Res | Frames | Steps | Step (ms) | DiT Gen (s) | Local CPU (ms) | Disagg RTT (ms) | Disagg Enc FPS | E2E Local (s) | E2E Disagg (s) | Cluster Videos/min (Disagg) | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `832x480` | `81` | `20` | 777.2 | 15.544 | 681.33 | 332.94 | 342.7 | 16.225 | 15.877 | 30.23 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `1280x720` | `81` | `35` | 2023.06 | 70.807 | 1503.35 | 533.42 | 199.2 | 72.311 | 71.341 | 6.73 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `1280x720` | `189` | `35` | 5837.69 | 204.319 | 3200.24 | 1012.97 | 260.1 | 207.519 | 205.332 | 2.34 | `PASS` |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `640x480` | `17` | `30` | 193.09 | 5.793 | 213.34 | 127.94 | 152.8 | 6.006 | 5.921 | 81.07 | `PASS` |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `832x480` | `61` | `30` | 611.74 | 18.352 | 545.24 | 245.63 | 300.0 | 18.897 | 18.598 | 25.81 | `PASS` |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `832x480` | `81` | `20` | 782.19 | 15.644 | 678.43 | 283.77 | 355.3 | 16.322 | 15.928 | 30.14 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `1280x720` | `81` | `35` | 1867.74 | 65.371 | 1493.78 | 605.87 | 198.4 | 66.865 | 65.977 | 7.28 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `1280x720` | `189` | `35` | 5183.35 | 181.417 | 3152.16 | 1161.8 | 258.3 | 184.57 | 182.579 | 2.63 | `PASS` |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `640x480` | `17` | `30` | 153.05 | 4.592 | 212.14 | 153.96 | 153.9 | 4.804 | 4.746 | 101.14 | `PASS` |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `4` | `8` | `832x480` | `61` | `30` | 621.55 | 18.646 | 545.66 | 222.78 | 314.5 | 19.192 | 18.869 | 25.44 | `PASS` |

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
