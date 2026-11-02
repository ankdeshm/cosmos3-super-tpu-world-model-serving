# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x` — Slice `v5p-32`

> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Stage 1 & Stage 2 TPU Compute (`torchtpu-mot-eager` / `torchtpu-mot-compiled`):** **Local-Shard Layer-Scaled MoT Operator Microbenchmark** (`run_child_torchtpu_mot_worker` in [`scripts/run_cosmos3_super_tpu_bench.py`](../../scripts/run_cosmos3_super_tpu_bench.py)) loading `129 GiB / tp_size` real `nvidia/Cosmos3-Super` `safetensors` shards onto HBM and timing `8` measured World Reasoner layers (`× 8 -> 64L`) plus `6` measured World Generator DiT layers (`× 6 -> 36L`) per host **excluding inter-chip TP `AllReduce` / Ulysses `AllToAll` collectives**. *Archived `torchtpu-mot-compiled` `B=1` note:* In sweeps recorded prior to the worker-thread warmup fix in `scripts/run_cosmos3_super_tpu_bench.py`, `B=1` decode `TPOT` (`483–1,971 ms`) included first-call thread-local `TorchDynamo` compilation inside `ThreadPoolExecutor`, whereas `B=2 / B=4` (`9.96–71.85 ms`) reflects steady-state compiled execution.
> - **Stage 2 Media Muxing (`POST /encode`):** **100% Real End-to-End HTTP + Multi-Threaded `ffmpeg` (`libx264` + `48 kHz` Stereo `AAC`) Video/Audio Muxing** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44` AVX-512), delivering **`2.39x–3.20x` measured muxing speedup** over TPU host CPUs.

### Slice `v5p-32` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-32/SUMMARY.md`](./SUMMARY.md) · [`results/v5p-32/summary.json`](./summary.json) *(updated `2026-11-02T19:06:01.483749+00:00`)*

#### Stage 1 — World Reasoner Long-Context Serving (`4K` – `128K`)

| Engine | Precision | `TP × Cluster DP` | Context | Batch | TTFT P50 (s) | TPOT P50 (ms) | Replica Prefill (tok/s) | Cluster Prefill (tok/s) | Cluster Decode (tok/s) |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `4,096` | `1` | `0.716` | **`429.32`** | `5,724.6` | **`45,796.8`** | **`18.64`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `4,096` | `4` | `2.709` | **`121.07`** | `6,048.6` | **`48,388.8`** | **`264.32`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `16,384` | `1` | `2.988` | **`117.63`** | `5,482.5` | **`43,860.0`** | **`68.00`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `16,384` | `2` | `5.585` | **`101.17`** | `5,866.9` | **`46,935.2`** | **`158.16`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `65,536` | `1` | `13.530` | **`122.30`** | `4,843.8` | **`38,750.4`** | **`65.44`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `65,536` | `2` | `26.813` | **`123.84`** | `4,888.3` | **`39,106.4`** | **`129.20`** |
| `torchtpu-mot-eager` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `131,072` | `1` | `34.813` | **`83.84`** | `3,765.0` | **`30,120.0`** | **`95.44`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `4,096` | `1` | `0.353` | **`717.86`** | `11,589.4` | **`92,715.2`** | **`11.12`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `4,096` | `4` | `1.360` | **`33.02`** | `12,050.8` | **`96,406.4`** | **`969.04`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `16,384` | `1` | `1.631` | **`711.27`** | `10,044.6` | **`80,356.8`** | **`11.28`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `16,384` | `2` | `3.038` | **`37.97`** | `10,784.6` | **`86,276.8`** | **`421.36`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `65,536` | `1` | `8.774` | **`742.23`** | `7,469.6` | **`59,756.8`** | **`10.80`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `65,536` | `2` | `17.810` | **`70.47`** | `7,359.5` | **`58,876.0`** | **`227.04`** |
| `torchtpu-mot-compiled` | `FP8_WEIGHT_DYNAMIC` | `TP=4 × DP=8` | `131,072` | `1` | `25.647` | **`1,829.22`** | `5,110.6` | **`40,884.8`** | **`4.40`** |

#### Stage 2 — World Generator Video + `48 kHz` Stereo Audio + `29-DoF` Action Rollouts

| Engine | Workload Mode | Res & Frames | Steps | Packed Tokens | Step (ms) | TPU Gen (s) | On-Host Mux (ms) | Disagg CPU Mux (ms) | Disagg FPS | E2E Disagg (s) | Cluster Videos/min |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `torchtpu-mot-eager` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `777.20` | `15.544` | `681.3` | **`332.9`** | `342.7` | **`15.877`** | **`30.23`** |
| `torchtpu-mot-eager` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `2,023.06` | `70.807` | `1,503.3` | **`533.4`** | `199.2` | **`71.341`** | **`6.73`** |
| `torchtpu-mot-eager` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,837.69` | `204.319` | `3,200.2` | **`1,013.0`** | `260.1` | **`205.332`** | **`2.34`** |
| `torchtpu-mot-eager` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `193.09` | `5.793` | `213.3` | **`127.9`** | `152.8` | **`5.921`** | **`81.07`** |
| `torchtpu-mot-eager` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `611.74` | `18.352` | `545.2` | **`245.6`** | `300.0` | **`18.598`** | **`25.81`** |
| `torchtpu-mot-compiled` | `T2V-480p-81f-20step` | `832x480` (`81f`) | `20` | `8,704` | `782.19` | `15.644` | `678.4` | **`283.8`** | `355.3` | **`15.928`** | **`30.14`** |
| `torchtpu-mot-compiled` | `T2VS-720p-81f-35step-StereoAudio` | `1280x720` (`81f`) | `35` | `19,200` | `1,867.74` | `65.371` | `1,493.8` | **`605.9`** | `198.4` | **`65.977`** | **`7.28`** |
| `torchtpu-mot-compiled` | `T2VS-720p-189f-35step-Default8s` | `1280x720` (`189f`) | `35` | `43,008` | `5,183.35` | `181.417` | `3,152.2` | **`1,161.8`** | `258.3` | **`182.579`** | **`2.63`** |
| `torchtpu-mot-compiled` | `ActionFD-AgiBot29D-480p-17f-30step` | `640x480` (`17f`) | `30` | `2,048` | `153.05` | `4.592` | `212.1` | **`154.0`** | `153.9` | **`4.746`** | **`101.14`** |
| `torchtpu-mot-compiled` | `ActionID-AV9D-480p-61f-30step` | `832x480` (`61f`) | `30` | `6,912` | `621.55` | `18.646` | `545.7` | **`222.8`** | `314.5` | **`18.869`** | **`25.44`** |
