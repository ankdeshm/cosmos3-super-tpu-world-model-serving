# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x` — Slice `v7x-8`

### Slice `v7x-8` (`v7x`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./SUMMARY.md) · [`results/v7x-8/summary.json`](./summary.json) *(updated `2026-11-02T20:01:33.948450+00:00`)*

#### Stage 1 — World Reasoner Long-Context Serving (`4K` – `128K`)

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT P50 (s) | TPOT P50 (ms) | Replica Prefill (tok/s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `4,096` | `1` | `0.074` | **`267.32`** | `55,723.0` | **`111,446.0`** | **`7.48`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `4,096` | `4` | `0.194` | **`50.66`** | `84,603.4` | **`169,206.8`** | **`157.92`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `16,384` | `1` | `0.309` | **`53.07`** | `53,087.4` | **`106,174.8`** | **`37.68`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `16,384` | `2` | `0.609` | **`49.66`** | `53,817.6` | **`107,635.2`** | **`80.54`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `65,536` | `1` | `2.751` | **`49.08`** | `23,821.1` | **`47,642.2`** | **`40.76`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `65,536` | `2` | `5.831` | **`52.09`** | `22,477.7` | **`44,955.4`** | **`76.78`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `131,072` | `1` | `9.684` | **`52.96`** | `13,534.3` | **`27,068.6`** | **`37.76`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `4,096` | `1` | `0.041` | **`483.20`** | `99,307.5` | **`198,615.0`** | **`4.14`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `4,096` | `4` | `0.146` | **`9.96`** | `112,485.1` | **`224,970.2`** | **`803.28`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `16,384` | `1` | `0.237` | **`496.07`** | `69,046.6` | **`138,093.2`** | **`4.04`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `16,384` | `2` | `0.501` | **`13.53`** | `65,438.7` | **`130,877.4`** | **`295.58`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `65,536` | `1` | `2.456` | **`509.65`** | `26,679.7` | **`53,359.4`** | **`3.92`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `65,536` | `2` | `4.994` | **`32.05`** | `26,245.4` | **`52,490.8`** | **`124.82`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=8 × DP=2` | `131,072` | `1` | `9.002` | **`1,576.43`** | `14,561.0` | **`29,122.0`** | **`1.26`** |

#### Stage 2 — World Generator Video + `48 kHz` Stereo Audio + `29-DoF` Action Rollouts

| Engine | Workload Mode | Res & Frames | Steps | Packed Tokens | Step (ms) | TPU Gen (s) | On-Host Mux (ms) | Disagg CPU Mux (ms) | Disagg FPS | E2E Disagg (s) | Cluster Videos/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `119.73` | `2.395` | `535.7` | **`307.6`** | `342.2` | **`2.702`** | **`44.41`** |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `389.57` | `13.635` | `1,135.1` | **`557.4`** | `197.9` | **`14.192`** | **`8.46`** |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `1,774.61` | `62.111` | `2,327.4` | **`995.7`** | `256.0` | **`63.107`** | **`1.90`** |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `29.71` | `0.891` | `160.7` | **`115.2`** | `159.6` | **`1.006`** | **`119.28`** |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `100.47` | `3.014` | `414.1` | **`220.6`** | `313.6` | **`3.235`** | **`37.09`** |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `101.93` | `2.039` | `516.3` | **`283.5`** | `351.7` | **`2.322`** | **`51.68`** |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `295.32` | `10.336` | `1,184.3` | **`476.9`** | `203.8` | **`10.813`** | **`11.10`** |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `1,354.78` | `47.417` | `2,334.7` | **`855.1`** | `274.2` | **`48.273`** | **`2.49`** |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `19.01` | `0.570` | `161.3` | **`121.1`** | `155.9` | **`0.691`** | **`173.66`** |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `83.01` | `2.490` | `414.6` | **`220.9`** | `313.0` | **`2.711`** | **`44.26`** |
