# `nvidia/Cosmos3-Super` (64.6B MoT World Reasoner & Generator) — `v7x-spot-4c` (`v7x`)

- **Updated (UTC):** `2026-11-06T12:44:44.023920+00:00`
- **Hardware:** `1` hosts x `8` TPU cores/host = `8` total TPU cores
- **Staged Checkpoint:** `30` shards (`120.35 GiB`)
- **Disaggregated CPU Offload Health:** `ok`

## 1. Omnimodal World Reasoner (`4K`, `16K`, `64K`, `128K` Context Lengths)

Compares `torchtpu-mot-eager` against `torchtpu-mot-compiled` (`torch.compile(..., dynamic=False)`) and hardware FP8 MXU (`torch._scaled_mm`).

| Engine | Precision | FP8 MXU | TP | Cluster DP | Context Len | Batch/Conc | TTFT P50 (s) | TPOT P50 (ms) | E2E (s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) | Status |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 4,096 | `1` | 0.073 | 257.73 | 132.033 | 56173.2 | 3.88 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 4,096 | `4` | 0.195 | 34.91 | 18.069 | 83867.6 | 114.58 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 16,384 | `1` | 0.307 | 36.42 | 18.955 | 53367.2 | 27.46 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 16,384 | `2` | 0.609 | 35.11 | 18.584 | 53826.9 | 56.97 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 65,536 | `1` | 2.751 | 39.34 | 22.893 | 23825.6 | 25.42 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 65,536 | `2` | 5.83 | 40.91 | 26.776 | 22481.2 | 48.89 | `PASS` |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 131,072 | `1` | 9.684 | 40.59 | 30.464 | 13535.2 | 24.64 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 4,096 | `1` | 0.041 | 488.27 | 250.036 | 100054.5 | 2.05 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 4,096 | `4` | 0.146 | 9.48 | 5.002 | 112252.9 | 421.72 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 16,384 | `1` | 0.236 | 479.78 | 245.884 | 69511.6 | 2.08 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 16,384 | `2` | 0.499 | 12.73 | 7.017 | 65663.3 | 157.11 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 65,536 | `1` | 2.45 | 496.91 | 256.87 | 26749.2 | 2.01 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 65,536 | `2` | 4.985 | 32.41 | 21.58 | 26294.0 | 61.71 | `PASS` |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `SCALAR_SCALE` | `8` | `1` | 131,072 | `1` | 8.974 | 1502.95 | 778.483 | 14606.0 | 0.67 | `PASS` |

## 2. Omnimodal Video + 48 kHz Stereo Audio + Robot Action Generation & CPU Offload Ablation

Reports pure TPU DiT generation time alongside end-to-end latency with local TPU-host CPU `ffmpeg` encoding vs remote disaggregated AVX-512 CPU offload (`multimodal-cpu-offload-svc`).

| Engine | Workload | Precision | Ulysses/TP | Cluster DP | Res | Frames | Steps | Step (ms) | DiT Gen (s) | Local CPU (ms) | Disagg RTT (ms) | Disagg Enc FPS | E2E Local (s) | E2E Disagg (s) | Cluster Videos/min (Disagg) | Status |
|---|---|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `832x480` | `81` | `20` | 120.78 | 2.416 | 545.84 | 331.34 | 354.1 | 2.961 | 2.747 | 21.84 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `1280x720` | `81` | `35` | 389.58 | 13.635 | 1140.35 | 549.89 | 194.8 | 14.776 | 14.185 | 4.23 | `PASS` |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `1280x720` | `189` | `35` | 1774.86 | 62.12 | 2347.11 | 1040.61 | 258.0 | 64.467 | 63.161 | 0.95 | `PASS` |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `640x480` | `17` | `30` | 28.19 | 0.846 | 178.15 | 115.69 | 157.7 | 1.024 | 0.961 | 62.43 | `PASS` |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `832x480` | `61` | `30` | 99.98 | 2.999 | 419.73 | 225.99 | 314.4 | 3.419 | 3.225 | 18.6 | `PASS` |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `832x480` | `81` | `20` | 101.66 | 2.033 | 514.74 | 329.54 | 346.0 | 2.548 | 2.363 | 25.39 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `1280x720` | `81` | `35` | 295.49 | 10.342 | 1111.74 | 563.52 | 197.1 | 11.454 | 10.906 | 5.5 | `PASS` |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `1280x720` | `189` | `35` | 1352.84 | 47.349 | 2377.81 | 1093.38 | 256.8 | 49.727 | 48.443 | 1.24 | `PASS` |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `640x480` | `17` | `30` | 18.47 | 0.554 | 161.65 | 124.79 | 152.5 | 0.716 | 0.679 | 88.37 | `PASS` |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `FP8_WEIGHT_DYNAMIC` | `8` | `1` | `832x480` | `61` | `30` | 83.39 | 2.502 | 411.19 | 221.17 | 310.9 | 2.913 | 2.723 | 22.03 | `PASS` |
