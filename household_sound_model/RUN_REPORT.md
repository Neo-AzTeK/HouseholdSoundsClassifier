# First training run — 2026-10-12

## Artifact

- Selected model: `artifacts/household_sound.pt` (324 KB; SHA-256 `3c48fb223f9299bca95e460c998442be912ea7381cdef31875284765a11857e9`)
- Original, unmixed baseline: `artifacts/household_sound_v1.pt` (SHA-256 `a783621a7fd6ff12cd0de69e5e75b1c4364ebf0990e33b30349e80526af6ed18`)
- Other candidates: `artifacts/household_sound_v2.pt` is the mixed model before its inference quiet gate was saved in the checkpoint; `artifacts/household_sound_v3.pt` used synthetic quiet training windows and scored worse on held-out speech.
- Metrics: `artifacts/household_sound.metrics.json`
- Dataset manifest: `data/manifest.jsonl` (SHA-256 `ccd9e76c2aa97f9e9cec1adf52816b2a10292e9f9486f92e89bdd885e155bc14`)
- Environment: Python 3.12.13, PyTorch 2.14.1+cpu, SciPy 1.18.1, SoundFile 0.14.0, NumPy 2.5.4.

## Training

Selected model command: `OMP_NUM_THREADS=8 ./.venv/bin/python train.py --epochs 25 --batch-size 32 --mix-probability 0.3 --output artifacts/household_sound_mixed.pt`

Original baseline command: `OMP_NUM_THREADS=8 ./.venv/bin/python train.py --epochs 25 --batch-size 32 --mix-probability 0`

The manifest has 1,194 train, 238 validation, and 233 held-out test clips. All 1,665 files opened as mono WAVs. The selected checkpoint is epoch 18 by validation macro F1 (0.5726). During training, 30% of windows mix a second public clip with a different label to teach overlapping sounds. The model receives one-second, 4,031-sample windows after simulating the Thingy FIR, decimation, high-pass filter, and 8-bit companding. A compact CNN produces five independent sigmoid scores.

## Held-out results

The thresholds were chosen on the validation split, then applied unchanged to the test split. One loud one-second window was scored per clip. The source labels describe entire clips, so these are proxy window metrics.

| Class | Precision | Recall | F1 | Positive test clips |
| --- | ---: | ---: | ---: | ---: |
| Voices | 0.647 | 0.489 | 0.557 | 45 |
| Machines | 0.536 | 0.357 | 0.429 | 42 |
| Collisions | 0.470 | 0.705 | 0.564 | 44 |
| Steps | 0.386 | 0.550 | 0.454 | 40 |
| Music | 0.647 | 0.524 | 0.579 | 42 |

The mean test F1 is 0.516, essentially tied with the original baseline's 0.515. On 20 deterministic, equal-loudness mixtures of held-out voice and music clips, the selected model put both classes above its validation thresholds in 7 cases; the original did so in 0. This is a small synthetic check, not a reliable estimate for room mixtures. These scores show that each class has learned some signal, with substantial errors remaining. The machine examples emphasize drills and power tools; dryers and washing machines have not been separately validated. Whole-room behavior still needs target-room recordings.

The inference adapter returns five zeros when the input RMS is below 0.0005. This prevents false scores on digital silence and very faint noise. It gates 2 of the 233 test windows and leaves the table's F1 values unchanged. A separately trained quiet-window candidate reduced raw silence scores but dropped mean test F1 to 0.488, so it was not selected. The gate threshold still needs testing with the real Thingy stream in the target room.

## Inference checks

- `./.venv/bin/python predict.py file data/audio/eval/101933.wav --summary` returned a five-score JSON object.
- `predict.py live --node-id 0` accepted streamed decoded samples and emitted two five-score objects from a 7,000-sample input. It also accepted companded `codes` and emitted five zeros for a digital-silence stream.
- On this laptop CPU, warmed one-window inference took a median 1.39 ms and p95 2.16 ms over 50 repetitions. This excludes microphone transport, the first one-second buffer, and Pi execution.

## Next validation

Record short examples of each class and quiet/background sound in the target room. Keep those recordings out of training for now; compare their scores and check whether voices and music can both rise when appropriate. The existing gateway still needs a bridge to the `predict.py live` NDJSON input. Hailo-10H deployment requires model conversion to HEF and Pi testing.
