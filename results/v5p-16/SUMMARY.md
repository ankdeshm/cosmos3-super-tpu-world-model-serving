# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x` — Slice `v5p-16`

### Slice `v5p-16` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-16/summary.json`](./summary.json) *(updated `2026-11-02T19:48:07.570541+00:00`)*

#### Stage 1 — World Reasoner Long-Context Serving (`4K` – `128K`)

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT P50 (s) | TPOT P50 (ms) | Replica Prefill (tok/s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | `0.714` | **`388.58`** | `5,733.6` | **`22,934.4`** | **`10.28`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | `2.708` | **`88.49`** | `6,049.4` | **`24,197.6`** | **`180.80`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | `2.985` | **`87.87`** | `5,489.2` | **`21,956.8`** | **`45.52`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | `5.580` | **`88.75`** | `5,872.8` | **`23,491.2`** | **`90.12`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | `13.513` | **`67.18`** | `4,850.0` | **`19,400.0`** | **`59.56`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | `26.761` | **`85.46`** | `4,897.8` | **`19,591.2`** | **`93.60`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | `34.706` | **`70.54`** | `3,776.7` | **`15,106.8`** | **`56.72`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `1` | `0.354` | **`709.93`** | `11,557.6` | **`46,230.4`** | **`5.64`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `4,096` | `4` | `1.359` | **`32.74`** | `12,056.8` | **`48,227.2`** | **`488.72`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `1` | `1.629` | **`697.40`** | `10,056.3` | **`40,225.2`** | **`5.72`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `16,384` | `2` | `3.037` | **`39.05`** | `10,790.4` | **`43,161.6`** | **`204.84`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `1` | `8.760` | **`716.10`** | `7,481.1` | **`29,924.4`** | **`5.60`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `65,536` | `2` | `17.784` | **`70.82`** | `7,370.3` | **`29,481.2`** | **`112.96`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=4` | `131,072` | `1` | `25.608` | **`1,971.29`** | `5,118.3` | **`20,473.2`** | **`2.04`** |

#### Stage 2 — World Generator Video + `48 kHz` Stereo Audio + `29-DoF` Action Rollouts

| Engine | Workload Mode | Res & Frames | Steps | Packed Tokens | Step (ms) | TPU Gen (s) | On-Host Mux (ms) | Disagg CPU Mux (ms) | Disagg FPS | E2E Disagg (s) | Cluster Videos/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `777.04` | `15.541` | `759.9` | **`321.8`** | `346.4` | **`15.863`** | **`15.13`** |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `2,022.23` | `70.778` | `1,494.4` | **`603.1`** | `197.8` | **`71.381`** | **`3.36`** |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,833.83` | `204.184` | `3,135.3` | **`910.2`** | `262.0` | **`205.094`** | **`1.17`** |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `192.88` | `5.786` | `212.3` | **`116.5`** | `156.4` | **`5.903`** | **`40.66`** |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `610.84` | `18.325` | `548.3` | **`222.7`** | `309.3` | **`18.548`** | **`12.94`** |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `781.37` | `15.627` | `683.6` | **`286.9`** | `351.1` | **`15.914`** | **`15.08`** |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `1,867.12` | `65.349` | `1,498.0` | **`486.3`** | `200.7` | **`65.835`** | **`3.65`** |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,180.30` | `181.310` | `3,137.7` | **`860.9`** | `270.8` | **`182.171`** | **`1.32`** |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `152.42` | `4.573` | `213.5` | **`121.1`** | `155.6` | **`4.694`** | **`51.13`** |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `622.41` | `18.672` | `548.7` | **`244.8`** | `313.0` | **`18.917`** | **`12.69`** |
