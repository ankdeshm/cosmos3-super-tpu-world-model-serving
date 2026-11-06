# `nvidia/Cosmos3-Super` (64.6B MoT World Reasoner & Generator) — `v7x-8` (`v7x`)

- **Updated (UTC):** `2026-11-06T12:52:10.348737+00:00`
- **Hardware:** `2` hosts x `8` TPU cores/host = `16` total TPU cores
- **Staged Checkpoint:** `30` shards (`120.35 GiB`)
- **Disaggregated CPU Offload Health:** `ok`

## 1. Omnimodal World Reasoner (`4K`, `16K`, `64K`, `128K` Context Lengths)

Compares `torchtpu-mot-eager` against `torchtpu-mot-compiled` (`torch.compile(..., dynamic=False)`) and hardware FP8 MXU (`torch._scaled_mm`).

| Engine | Precision | FP8 MXU | TP | Cluster DP | Context Len | Batch/Conc | TTFT P50 (s) | TPOT P50 (ms) | E2E (s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 4,096 | `1` | 0.071 | 257.43 | 131.877 | 115270.8 | 7.76 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 4,096 | `4` | 0.192 | 34.91 | 18.065 | 170378.0 | 229.18 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 16,384 | `1` | 0.306 | 36.46 | 18.975 | 107191.6 | 54.84 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 16,384 | `2` | 0.604 | 40.79 | 21.49 | 108461.2 | 98.06 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 65,536 | `1` | 2.741 | 35.72 | 21.03 | 47815.4 | 56.0 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 65,536 | `2` | 5.815 | 40.9 | 26.758 | 45081.4 | 97.78 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 131,072 | `1` | 9.649 | 50.45 | 35.478 | 27169.2 | 39.64 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 4,096 | `1` | 0.041 | 503.77 | 257.973 | 197839.2 | 3.98 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 4,096 | `4` | 0.144 | 10.76 | 5.655 | 227462.8 | 743.22 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 16,384 | `1` | 0.236 | 498.2 | 255.316 | 139126.4 | 4.02 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 16,384 | `2` | 0.499 | 13.32 | 7.316 | 131434.8 | 300.4 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 65,536 | `1` | 2.441 | 548.03 | 283.032 | 53687.8 | 3.64 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 65,536 | `2` | 4.977 | 31.1 | 20.901 | 52674.8 | 128.6 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `2` | 131,072 | `1` | 8.946 | 1602.3 | 829.322 | 29303.8 | 1.24 | `PASS` |

## 2. Omnimodal Video + 48 kHz Stereo Audio + Robot Action Generation & CPU Offload Ablation

Reports pure TPU DiT generation time alongside end-to-end latency with local TPU-host CPU `ffmpeg` encoding vs remote disaggregated AVX-512 CPU offload (`multimodal-cpu-offload-svc`).

| Engine | Workload | Precision | Ulysses/TP | Cluster DP | Res | Frames | Steps | Step (ms) | DiT Gen (s) | Local CPU (ms) | Disagg RTT (ms) | Disagg Enc FPS | E2E Local (s) | E2E Disagg (s) | Cluster Videos/min (Disagg) | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `832x480` | `81` | `20` | 117.49 | 2.35 | 519.8 | 265.87 | 353.2 | 2.87 | 2.616 | 45.87 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `1280x720` | `81` | `35` | 390.28 | 13.66 | 1118.71 | 476.47 | 200.1 | 14.779 | 14.136 | 8.49 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `1280x720` | `189` | `35` | 1771.6 | 62.006 | 2501.7 | 857.6 | 274.0 | 64.508 | 62.863 | 1.91 | `PASS` |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `640x480` | `17` | `30` | 33.89 | 1.017 | 169.7 | 115.0 | 155.8 | 1.186 | 1.132 | 106.01 | `PASS` |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `832x480` | `61` | `30` | 101.21 | 3.036 | 421.11 | 217.07 | 315.6 | 3.458 | 3.253 | 36.89 | `PASS` |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `832x480` | `81` | `20` | 102.43 | 2.049 | 528.72 | 285.56 | 348.2 | 2.577 | 2.334 | 51.41 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `1280x720` | `81` | `35` | 295.83 | 10.354 | 1121.56 | 477.26 | 199.1 | 11.476 | 10.831 | 11.08 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `1280x720` | `189` | `35` | 1352.06 | 47.322 | 2412.05 | 857.17 | 271.1 | 49.734 | 48.179 | 2.49 | `PASS` |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `640x480` | `17` | `30` | 19.31 | 0.579 | 164.03 | 120.21 | 157.1 | 0.743 | 0.699 | 171.67 | `PASS` |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `2` | `832x480` | `61` | `30` | 83.19 | 2.496 | 415.06 | 222.55 | 305.4 | 2.911 | 2.718 | 44.15 | `PASS` |
