# `nvidia/Cosmos3-Super` (64.6B MoT World Model) Serving & Benchmarking on Cloud TPU `v5p` & `v7x`

nvidia/Cosmos3-Super (64.6B Mixture-of-Transformers Omnimodal World Reasoner & Video/Audio/Action Generator) Serving on Cloud TPU v5p & v7x with Disaggregated CPU Offloading.


> [!IMPORTANT]
> **Benchmark Scope & Measurement Methodology**
> - **Stage 1 & Stage 2 TPU Compute (`torchtpu-mot-eager` / `torchtpu-mot-compiled`):** **Local-Shard Layer-Scaled MoT Operator Microbenchmark** (`run_child_torchtpu_mot_worker` in [`scripts/run_cosmos3_super_tpu_bench.py`](./scripts/run_cosmos3_super_tpu_bench.py)) loading `129 GiB / tp_size` real `nvidia/Cosmos3-Super` `safetensors` shards onto HBM and timing `8` measured World Reasoner layers (`× 8 -> 64L`) plus `6` measured World Generator DiT layers (`× 6 -> 36L`) per host **excluding inter-chip TP `AllReduce` / Ulysses `AllToAll` collectives**. *Archived `torchtpu-mot-compiled` `B=1` note:* In sweeps recorded prior to the worker-thread warmup fix in `scripts/run_cosmos3_super_tpu_bench.py`, `B=1` decode `TPOT` (`483–1,971 ms`) included first-call thread-local `TorchDynamo` compilation inside `ThreadPoolExecutor`, whereas `B=2 / B=4` (`9.96–71.85 ms`) reflects steady-state compiled execution.
> - **Stage 2 Media Muxing (`POST /encode`):** **100% Real End-to-End HTTP + Multi-Threaded `ffmpeg` (`libx264` + `48 kHz` Stereo `AAC`) Video/Audio Muxing** against `multimodal-cpu-offload-svc:8090` (`c3-standard-44` AVX-512), delivering **`2.39x–3.20x` measured muxing speedup** over TPU host CPUs.

---

## 1. Model & System Architecture

| Component | Specification |
|---|---|
| **Checkpoint** | [`nvidia/Cosmos3-Super`](https://huggingface.co/nvidia/Cosmos3-Super) (`64.6B` BF16 / FP8) |
| **Backbone** | Dual-Tower Mixture-of-Transformers (MoT): World Reasoner (`4K / 16K / 64K / 128K` context) + Flow-Matching World Generator |
| **Output Modalities** | Synchronized World Video (`480p / 720p`) + `48 kHz` Stereo Audio + `29-DoF` Continuous Robot Action Trajectories |
| **TPU Execution** | `TorchTPU` (`eager` & `torch.compile(..., dynamic=False)`), `BF16` & `FP8_WEIGHT_DYNAMIC` (`out_dtype=bfloat16`), `TP=4` / `TP=8` |
| **Disaggregated CPU Offload** | `multimodal-cpu-offload-svc` (`c3-standard-44` AVX-512, `HTTP/1.1 Keep-Alive + TCP_NODELAY`): `POST /encode` H.264 MP4 + `48 kHz` AAC muxing & `POST /preprocess_vision` |

---

## 2. Disaggregated CPU Offloading Architecture (`multimodal-cpu-offload-svc`)

Non-TPU media muxing, high-resolution image compression, and multi-camera/multi-frame vision preprocessing are offloaded over intra-cluster `ClusterIP` HTTP (`port 8090`) to dedicated `c3-standard-44` (`44` Sapphire Rapids vCPUs, AVX-512, `176 GiB` DDR5) CPU pods ([`scripts/multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) & [`scripts/cpu_offload_client.py`](./scripts/cpu_offload_client.py)), keeping TPU host CPUs dedicated to XLA/PJRT dispatch:

- `POST /encode` — Multi-threaded `ffmpeg` (`libx264` + `48 kHz` stereo `AAC`) video/audio MP4 muxing.
- `POST /encode_image` — Thread-pooled `Pillow` `PNG` / `WebP` / `JPEG` high-resolution image encoding.
- `POST /encode_wav` — Vectorized `float32` / `int16` audio normalization + `RIFF/WAV` packet framing (`24 kHz` / `48 kHz`).
- `POST /preprocess_vision` — Multi-camera / multi-frame RGB decode, bicubic resize, and `bfloat16`/`float32` normalization.

---

## 3. Hardware Platforms Evaluated

| Accelerator | Slice Topologies | HBM Capacity & Bandwidth | Host Memory (`/dev/shm`) |
|---|---|---|---|
| **Cloud TPU `v5p`** | `v5p-16` (`2x2x4`, `4` hosts × `4` chips) & `v5p-32` (`2x4x4`, `8` hosts × `4` chips) | `95 GiB` HBM2e / chip (`2,765 GB/s` / chip, `459` BF16 TFLOP/s) | `462 GiB` RAM (`380 GiB` tmpfs) |
| **Cloud TPU `v7x` (Ironwood)** | `v7x-4` (`2x2x1`, `8` TensorCores), `v7x-8` (`2x2x2`, `16` TensorCores), & `v7x-16` (`2x2x4`, `32` TensorCores) | `192 GiB` HBM3e / chip (`96 GiB` / TensorCore, `7,370 GB/s` / chip, `2,307` BF16 / `4,614` FP8 TFLOP/s per chip) | `944 GiB` RAM (`850 GiB` tmpfs) |

---

## 4. Before vs. After Disaggregated CPU Offloading (`Keep-Alive + TCP_NODELAY`) & `torch.compile` Optimization

Direct A/B comparison across `v5p-32` (`TP=4 × DP=8`) and `v5p-16` (`Wave-3 Keep-Alive`) comparing **Before** (uncompiled `torchtpu-mot-eager` + synchronous on-host `ffmpeg` muxing) against **After** (`torchtpu-mot-compiled` + persistent `HTTP/1.1 Keep-Alive + TCP_NODELAY` disaggregated `c3-standard-44` AVX-512 `multimodal-cpu-offload-svc:8090` muxing):

| Stage & Workload | Before (On-Host CPU / Eager) | After (Disagg AVX-512 CPU + `torch.compile`) | Measured Gain | Code & Result References |
|---|---:|---:|---:|---|
| **World Reasoner `4K` (`B=4`) Decode TPOT** | `121.07 ms` (`264.3 tok/s` cluster) | **`33.02 ms`** (**`969.0 tok/s`** cluster) | **`3.67x` faster decode** | [`run_cosmos3_super_tpu_bench.py`](./scripts/run_cosmos3_super_tpu_bench.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **World Reasoner `16K` (`B=2`) Decode TPOT** | `101.17 ms` (`158.2 tok/s` cluster) | **`37.97 ms`** (**`421.4 tok/s`** cluster) | **`2.66x` faster decode** | [`run_cosmos3_super_tpu_bench.py`](./scripts/run_cosmos3_super_tpu_bench.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **World Reasoner `64K` (`B=2`) Decode TPOT** | `123.84 ms` (`129.2 tok/s` cluster) | **`70.47 ms`** (**`227.0 tok/s`** cluster) | **`1.76x` faster decode** | [`run_cosmos3_super_tpu_bench.py`](./scripts/run_cosmos3_super_tpu_bench.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **World Gen `480p 81f` `H.264 + 48kHz AAC` Mux** | `678.4 ms` on-host CPU | **`283.8 ms`** (`355.3 fps` on AVX-512 pod) | **`2.39x` faster mux** (`-394.6 ms`/video) | [`multimodal_cpu_offload_server.py`](./scripts/multimodal_cpu_offload_server.py) · [`v5p-32/summary.json`](./results/v5p-32/summary.json) |
| **World Gen `720p 81f` `Keep-Alive + TCP_NODELAY` Mux** | `1,475.6 – 1,493.8 ms` on-host CPU | **`556.1 ms`** (`204.7 fps`, Wave-3 Keep-Alive) | **`2.65x` faster mux** (`-919.5 ms`/video) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |
| **World Gen `720p 189f` LongHorizon `Keep-Alive` Mux** | `3,138.0 – 3,152.2 ms` on-host CPU | **`981.9 ms`** (`253.2 – 262.7 fps`, Wave-3 Keep-Alive) | **`3.20x` faster mux** (**`-2.16 s`/video**) | [`cpu_offload_client.py`](./scripts/cpu_offload_client.py) · [`v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) |

---

## 5. Complete Benchmark Tables (`v5p` & `v7x`)

Every table below is generated directly from the verified `PASS` records in [`results/`](./results/) with 1:1 links to raw `summary.json` artifacts:

### Slice `v5p-16` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16/SUMMARY.md`](./results/v5p-16/SUMMARY.md) · [`results/v5p-16/summary.json`](./results/v5p-16/summary.json) *(updated `2026-11-02T19:48:07.570541+00:00`)*

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

### Slice `v5p-16-wave3-keepalive` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave3-keepalive/SUMMARY.md`](./results/v5p-16-wave3-keepalive/SUMMARY.md) · [`results/v5p-16-wave3-keepalive/summary.json`](./results/v5p-16-wave3-keepalive/summary.json) *(updated `2026-11-02T20:15:10.266691+00:00`)*

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

### Slice `v5p-16-wave4-adaptive64k` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-16-wave4-adaptive64k/SUMMARY.md`](./results/v5p-16-wave4-adaptive64k/SUMMARY.md) · [`results/v5p-16-wave4-adaptive64k/summary.json`](./results/v5p-16-wave4-adaptive64k/summary.json) *(updated `2026-11-02T21:14:46.723898+00:00`)*

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

### Slice `v5p-32` (`v5p`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v5p-32/SUMMARY.md`](./results/v5p-32/SUMMARY.md) · [`results/v5p-32/summary.json`](./results/v5p-32/summary.json) *(updated `2026-11-02T19:06:01.483749+00:00`)*

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

### Slice `v7x-8` (`v7x`, `24` Verified Benchmark Points)

- **Raw Artifacts:** [`results/v7x-8/SUMMARY.md`](./results/v7x-8/SUMMARY.md) · [`results/v7x-8/summary.json`](./results/v7x-8/summary.json) *(updated `2026-11-02T20:01:33.948450+00:00`)*

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

---

## 6. Repository Layout & Reproduction

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
