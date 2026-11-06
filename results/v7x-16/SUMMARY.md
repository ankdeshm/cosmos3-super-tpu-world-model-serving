# `nvidia/Cosmos3-Super` (64.6B MoT World Reasoner & Generator) — `v7x-16` (`v7x`)

- **Updated (UTC):** `2026-11-03T09:29:39.791364+00:00`
- **Hardware:** `4` hosts x `8` TPU cores/host = `32` total TPU cores
- **Staged Checkpoint:** `30` shards (`120.35 GiB`)
- **Disaggregated CPU Offload Health:** `ok`

## 1. Omnimodal World Reasoner (`4K`, `16K`, `64K`, `128K` Context Lengths)

Compares `torchtpu-mot-eager` against `torchtpu-mot-compiled` (`torch.compile(..., dynamic=False)`) and hardware FP8 MXU (`torch._scaled_mm`).

| Engine | Precision | FP8 MXU | TP | Cluster DP | Context Len | Batch/Conc | TTFT P50 (s) | TPOT P50 (ms) | E2E (s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 4,096 | `1` | 0.073 | 258.39 | 132.367 | 224080.8 | 15.48 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 4,096 | `4` | 0.193 | 35.26 | 18.245 | 339101.6 | 453.8 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 16,384 | `1` | 0.307 | 35.06 | 18.257 | 213630.8 | 114.08 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 16,384 | `2` | 0.61 | 38.35 | 20.247 | 214968.8 | 208.56 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 65,536 | `1` | 2.757 | 50.9 | 28.817 | 95070.4 | 78.6 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 65,536 | `2` | 5.835 | 41.85 | 27.264 | 89857.2 | 191.16 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 131,072 | `1` | 9.701 | 41.17 | 30.781 | 54046.8 | 97.16 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 4,096 | `1` | 0.042 | 482.58 | 247.123 | 388090.8 | 8.28 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 4,096 | `4` | 0.146 | 8.76 | 4.629 | 449424.4 | 1827.4 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 16,384 | `1` | 0.237 | 488.85 | 250.531 | 276084.8 | 8.2 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 16,384 | `2` | 0.502 | 14.43 | 7.89 | 261269.6 | 554.36 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 65,536 | `1` | 2.46 | 509.85 | 263.502 | 106565.6 | 7.84 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 65,536 | `2` | 4.992 | 31.47 | 21.103 | 105026.0 | 254.24 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `4` | 131,072 | `1` | 9.015 | 1580.85 | 818.412 | 58158.4 | 2.52 | `PASS` |

## 2. Omnimodal Video + 48 kHz Stereo Audio + Robot Action Generation & CPU Offload Ablation

Reports pure TPU DiT generation time alongside end-to-end latency with local TPU-host CPU `ffmpeg` encoding vs remote disaggregated AVX-512 CPU offload (`multimodal-cpu-offload-svc`).

| Engine | Workload | Precision | Ulysses/TP | Cluster DP | Res | Frames | Steps | Step (ms) | DiT Gen (s) | Local CPU (ms) | Disagg RTT (ms) | Disagg Enc FPS | E2E Local (s) | E2E Disagg (s) | Cluster Videos/min (Disagg) | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `832x480` | `81` | `20` | 118.41 | 2.368 | 529.08 | 272.11 | 344.9 | 2.897 | 2.64 | 90.91 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `1280x720` | `81` | `35` | 390.19 | 13.657 | 1173.88 | 479.19 | 199.5 | 14.831 | 14.136 | 16.98 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `1280x720` | `189` | `35` | 1771.79 | 62.012 | 2382.22 | 881.37 | 265.6 | 64.395 | 62.894 | 3.82 | `PASS` |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `640x480` | `17` | `30` | 27.25 | 0.817 | 166.22 | 114.26 | 157.4 | 0.984 | 0.932 | 257.51 | `PASS` |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `832x480` | `61` | `30` | 100.19 | 3.006 | 415.32 | 220.19 | 311.7 | 3.421 | 3.226 | 74.4 | `PASS` |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `832x480` | `81` | `20` | 102.99 | 2.06 | 515.14 | 304.23 | 350.9 | 2.575 | 2.364 | 101.52 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `1280x720` | `81` | `35` | 296.55 | 10.379 | 1120.06 | 505.64 | 197.1 | 11.499 | 10.885 | 22.05 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `1280x720` | `189` | `35` | 1354.58 | 47.41 | 2318.44 | 972.11 | 273.2 | 49.729 | 48.382 | 4.96 | `PASS` |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `640x480` | `17` | `30` | 18.68 | 0.56 | 175.12 | 119.37 | 157.3 | 0.735 | 0.68 | 352.94 | `PASS` |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `4` | `832x480` | `61` | `30` | 83.56 | 2.507 | 412.78 | 238.85 | 313.3 | 2.92 | 2.746 | 87.4 | `PASS` |
