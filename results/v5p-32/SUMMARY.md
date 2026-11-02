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
