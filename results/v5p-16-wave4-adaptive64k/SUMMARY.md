# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x` — Slice `v5p-16-wave4-adaptive64k`

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./summary.json) *(updated `2026-11-02T21:14:46.723898+00:00`)*

#### Stage 1 — World Reasoner Long-Context Serving (`4K` – `128K`)

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT P50 (s) | TPOT P50 (ms) | Replica Prefill (tok/s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | `0.714` | **`404.23`** | `5,736.3` | **`22,945.2`** | **`9.88`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | `2.708` | **`75.36`** | `6,051.2` | **`24,204.8`** | **`212.32`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | `2.984` | **`80.65`** | `5,490.7` | **`21,962.8`** | **`49.60`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | `5.579` | **`65.02`** | `5,873.9` | **`23,495.6`** | **`123.04`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | `13.510` | **`64.99`** | `4,850.8` | **`19,403.2`** | **`61.56`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | `26.762` | **`89.15`** | `4,897.7` | **`19,590.8`** | **`89.72`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | `34.705` | **`67.54`** | `3,776.8` | **`15,107.2`** | **`59.24`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | `0.353` | **`703.45`** | `11,594.0` | **`46,376.0`** | **`5.68`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | `1.358` | **`32.59`** | `12,061.0` | **`48,244.0`** | **`490.88`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | `1.629` | **`694.14`** | `10,055.9` | **`40,223.6`** | **`5.76`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | `3.036` | **`37.70`** | `10,792.2` | **`43,168.8`** | **`212.20`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | `8.760` | **`714.76`** | `7,481.1` | **`29,924.4`** | **`5.60`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | `17.783` | **`71.18`** | `7,370.4` | **`29,481.6`** | **`112.40`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | `25.607` | **`1,906.60`** | `5,118.5` | **`20,474.0`** | **`2.08`** |

#### Stage 2 — World Generator Video + `48 kHz` Stereo Audio + `29-DoF` Action Rollouts

| Engine | Workload Mode | Res & Frames | Steps | Packed Tokens | Step (ms) | TPU Gen (s) | On-Host Mux (ms) | Disagg CPU Mux (ms) | Disagg FPS | E2E Disagg (s) | Cluster Videos/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `776.78` | `15.536` | `680.7` | **`291.7`** | `351.2` | **`15.827`** | **`15.16`** |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `2,021.43` | `70.750` | `1,499.7` | **`498.8`** | `204.2` | **`71.249`** | **`3.37`** |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,833.48` | `204.172` | `3,142.5` | **`962.9`** | `265.1` | **`205.135`** | **`1.17`** |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `191.32` | `5.740` | `214.1` | **`117.8`** | `155.4` | **`5.857`** | **`40.98`** |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `609.84` | `18.295` | `546.6` | **`232.7`** | `310.3` | **`18.528`** | **`12.95`** |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `781.47` | `15.629` | `682.3` | **`289.2`** | `349.6` | **`15.919`** | **`15.08`** |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `1,865.56` | `65.295` | `1,500.7` | **`478.1`** | `198.1` | **`65.773`** | **`3.65`** |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,180.38` | `181.313` | `3,129.2` | **`899.3`** | `265.9` | **`182.213`** | **`1.32`** |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `152.52` | `4.576` | `217.0` | **`116.7`** | `158.7` | **`4.692`** | **`51.15`** |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `620.54` | `18.616` | `551.9` | **`218.7`** | `314.8` | **`18.835`** | **`12.74`** |
