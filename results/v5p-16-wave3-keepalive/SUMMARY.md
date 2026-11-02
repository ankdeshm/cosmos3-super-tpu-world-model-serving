# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave3-keepalive`

### Slice `v5p-16-wave3-keepalive` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./summary.json) *(updated `2026-11-02T20:15:10.266691+00:00`)*

#### Stage 1 — World Reasoner Long-Context Serving (`4K` – `128K`)

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT P50 (s) | TPOT P50 (ms) | Replica Prefill (tok/s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | `0.714` | **`392.45`** | `5,739.9` | **`22,959.6`** | **`10.20`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | `2.708` | **`77.87`** | `6,050.5` | **`24,202.0`** | **`205.48`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | `2.984` | **`84.98`** | `5,490.9` | **`21,963.6`** | **`47.08`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | `5.579` | **`78.82`** | `5,873.3` | **`23,493.2`** | **`101.48`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | `13.511` | **`67.67`** | `4,850.5` | **`19,402.0`** | **`59.12`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | `26.760` | **`75.28`** | `4,898.1` | **`19,592.4`** | **`106.28`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | `34.704` | **`67.20`** | `3,776.9` | **`15,107.6`** | **`59.52`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | `0.354` | **`699.03`** | `11,586.7` | **`46,346.8`** | **`5.72`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | `1.359` | **`32.66`** | `12,056.5` | **`48,226.0`** | **`489.84`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | `1.630` | **`695.73`** | `10,053.2` | **`40,212.8`** | **`5.76`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | `3.037` | **`38.05`** | `10,789.7` | **`43,158.8`** | **`210.24`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | `8.761` | **`736.95`** | `7,480.7` | **`29,922.8`** | **`5.44`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | `17.784` | **`71.85`** | `7,370.2` | **`29,480.8`** | **`111.36`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | `25.608` | **`1,949.94`** | `5,118.3` | **`20,473.2`** | **`2.04`** |

#### Stage 2 — World Generator Video + `48 kHz` Stereo Audio + `29-DoF` Action Rollouts

| Engine | Workload Mode | Res & Frames | Steps | Packed Tokens | Step (ms) | TPU Gen (s) | On-Host Mux (ms) | Disagg CPU Mux (ms) | Disagg FPS | E2E Disagg (s) | Cluster Videos/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `777.20` | `15.544` | `757.5` | **`304.2`** | `346.1` | **`15.848`** | **`15.14`** |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `2,021.54` | `70.754` | `1,506.3` | **`510.7`** | `199.5` | **`71.264`** | **`3.37`** |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,833.40` | `204.169` | `3,138.0` | **`981.9`** | `253.2` | **`205.151`** | **`1.17`** |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `192.54` | `5.776` | `214.6` | **`114.1`** | `156.4` | **`5.890`** | **`40.75`** |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `610.61` | `18.318` | `559.7` | **`219.4`** | `311.0` | **`18.538`** | **`12.95`** |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `781.79` | `15.636` | `681.0` | **`338.1`** | `343.0` | **`15.974`** | **`15.02`** |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `1,865.79` | `65.303` | `1,475.6` | **`556.0`** | `204.7` | **`65.859`** | **`3.64`** |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,180.48` | `181.317` | `3,126.3` | **`1,033.4`** | `262.7` | **`182.350`** | **`1.32`** |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `152.79` | `4.584` | `213.4` | **`128.1`** | `155.9` | **`4.712`** | **`50.93`** |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `621.03` | `18.631` | `543.8` | **`235.4`** | `309.1` | **`18.866`** | **`12.72`** |
