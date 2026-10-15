# How this household sound model was trained

This describes the **current checkpoint**, `artifacts/household_sound.pt`, and how to repeat or extend the experiment. The [run report](RUN_REPORT.md) records the exact artifact hashes and measured results.

## What the model learns

The input is approximately one second of audio from one microphone. The output contains five independent scores between 0 and 1: `voices`, `machines`, `collisions`, `steps`, and `music`. Each training example has five binary targets, such as `[1, 0, 0, 0, 1]` for speech over music. This is **multi-label** classification: the scores do not have to add to one. They are model scores, not calibrated probabilities of a source being present.

The model learns statistical patterns in examples. It does not have a built-in definition of a washing machine or a footstep. Its ability to recognize either depends on the examples and labels it sees.

## 1. Select and label public clips

`dataset.py` reads the FSD50K development and evaluation metadata and maps source tags to our five classes. For example, `Speech`, `Conversation`, and `Singing` map to voices; `Walk_and_footsteps` maps to steps; `Music` maps to music. Machine tags currently include `Mechanical_fan`, `Microwave_oven`, `Printer`, `Power_tool`, and `Drill`. Impact tags include `Slam`, `Knock`, `Thump_and_thud`, `Shatter`, and some dropped-object sounds. Rain, wind, water, dog, bird, and insect examples provide selected clips with none of the five target labels. These tag lists are editable in `dataset.py` for later experiments.

The script uses a fixed seed and writes `data/manifest.jsonl`. Each row records the clip ID, original tags, mapped targets, split, source URL, and available uploader and license metadata. Our subset has **1,665 clips**: 1,194 train, 238 validation, and 233 test. A clip appears in only one split. FSD50K's evaluation partition supplies the test clips. The present audio subset is about 1.2 GB. See the [FSD50K dataset record](https://zenodo.org/records/4060432) for its source and license information.

The source labels describe entire clips. They do not say exactly *when* a sound occurs within a clip. That is a source of label noise when training on one-second windows.

## 2. Make training audio resemble the Thingy stream

`audio_model.py` loads each source clip as mono and simulates the current Nordic Thingy:53 audio path. It resamples to 16,125 samples/s, applies the firmware's FIR filter, keeps one of every four samples, applies a high-pass filter, and simulates the 8-bit logarithmic compression and decoding. The resulting stream is about **4,031 samples/s**. This step matters because a classifier trained on full-bandwidth public recordings might rely on frequencies and detail that the real stream cannot deliver. It is an approximation; real hardware and room effects still need testing.

The training code takes **4,031-sample windows**, approximately 1 second long. It finds the three loudest candidate windows in each clip and randomly chooses among them during training. Validation and test use the loudest one. Choosing active windows reduces cases where a clip has a speech tag but the sampled second contains silence; it cannot guarantee the labeled sound is in that second. Processed clips are cached in `data/cache/` so future epochs do not repeat the simulation.

Each window becomes a log-magnitude spectrogram with **129 frequency bins and 48 time frames**. The STFT uses a 256-sample FFT and 80-sample hop. This is a compact image of how frequency content changes across the second.

### Why this feature representation?

The decoded stream is sampled at 4,031.25 Hz, so it contains usable frequencies only up to about **2,016 Hz**. A 256-sample Fourier frame is about **63.5 ms** long and separates frequencies by about **15.7 Hz**. Advancing 80 samples places a frame about every **19.8 ms**. Those settings make a 1-second input into a `129 × 48` map: 129 bins from 0 Hz to Nyquist, across 48 time positions. The choice balances time detail for knocks and steps against frequency detail for hums and voiced sounds. Changing the FFT size would trade one for the other.

We keep the magnitude of each complex Fourier value, then calculate `log(1 + 1000 × magnitude)`. Magnitude discards phase, which is less useful for this first sound-category task. The logarithm compresses the large gap between loud and quiet spectral components so quieter harmonics remain visible alongside strong peaks. The factor 1000 sets where that compression becomes noticeable; it is a fixed scale, **not a learned normalization**. The model currently uses linear frequency bins, with no mel filterbank, MFCCs, or per-recording standardization. Those are reasonable alternatives to compare later, but this run did not compare them experimentally.

An impact should appear as a short vertical spread of energy; a steady motor can appear as persistent horizontal bands; speech and music form changing patterns. These are useful intuitions for the feature choice, not rules the network explicitly follows. At this low sample rate, higher harmonics and some bright impact cues are absent before the spectrogram is made. Better feature extraction cannot recover frequencies missing from the Thingy stream.

## 3. Train the neural network

`SoundCNN` in `audio_model.py` has four convolution, batch-normalization, activation, and pooling blocks, followed by global average pooling and five output units. It has **79,589 learned parameters**. It starts with random weights; there are no pretrained weights in this checkpoint.

### Exact network layout

The tensor shape below is `channels × frequency positions × time positions`; the batch dimension is omitted. Every convolution uses a `3 × 3` kernel, one cell of padding on each side, and stride 1. Every max-pool uses `2 × 2` with stride 2.

| Stage | Operations | Output shape | Learned parameters |
| --- | --- | ---: | ---: |
| Input | Log spectrogram | `1 × 129 × 48` | 0 |
| Block 1 | Conv 1→16, BatchNorm, ReLU, MaxPool | `16 × 64 × 24` | 192 |
| Block 2 | Conv 16→32, BatchNorm, ReLU, MaxPool | `32 × 32 × 12` | 4,704 |
| Block 3 | Conv 32→64, BatchNorm, ReLU, MaxPool | `64 × 16 × 6` | 18,624 |
| Block 4 | Conv 64→96, BatchNorm, ReLU, MaxPool | `96 × 8 × 3` | 55,584 |
| Pool and head | Average each of 96 maps; Linear 96→5 | `5` logits | 485 |

The parameter totals include convolution biases and BatchNorm's learned scale and offset. They sum to **79,589**.

Each convolution learns a small time-frequency pattern. Early filters can respond to local edges, bands, or bursts; later filters combine patterns over a wider area. `ReLU` lets the blocks represent nonlinear combinations. `BatchNorm` rescales activations during training and stores running statistics for inference. Max-pooling reduces each spatial dimension by about half, shrinking compute and making exact event position less important. With four pools, the last map has only **three time positions**; this is economical but may blur the timing of very short impacts or footstep sequences.

Global average pooling reduces each of the 96 final maps to one number. The linear head combines those numbers into five logits. This avoids a large dense layer tied to every map position, keeping the checkpoint small and inference cheap. It also means the final decision emphasizes whether patterns occur in the window more than precisely *when* they occur. One second gives the model time to observe a step or repeated machine pattern, while creating roughly a one-second initial collection delay. Overlapping windows then refresh the output every half second.

The architecture was chosen as a compact **baseline** for a modest public subset and eventual edge deployment. The parameter count and local CPU timing show that it is small; they do **not** establish that it is optimal for the Pi, Hailo compiler, or these five sounds. We have not run an architecture comparison against mel-spectrogram CNNs, pretrained audio encoders, or models with more temporal detail.

### Other design choices to compare

The current CNN treats the spectrogram as a two-dimensional array, much like an image. That is a valid audio baseline, but the axes have different meanings: time has an order and duration, while frequency has pitch and physical units. Sharing the same small convolution across every frequency may miss the importance of *where* a pattern occurs. Pooling time four times and then averaging can hide a short collision. These are design tradeoffs, not proven causes of the current errors.

| Candidate | What it changes | Why it may help | Cost or uncertainty |
| --- | --- | --- | --- |
| Log-mel or PCEN features | Replace the current linear-frequency log spectrum | Fewer frequency bins; PCEN can adapt to persistent background | May remove useful narrow hum bands; must be measured on our stream |
| Keep more time resolution | Pool frequency more than time; use temporal max/attention pooling | Brief impacts and separate footsteps remain distinguishable | More computation; global clip labels still give imprecise event times |
| CNN plus temporal layer | Add a small temporal convolution or recurrent layer after frequency processing | Learn whether patterns persist, repeat, or occur briefly | More complexity and state/latency to validate |
| Pretrained audio encoder | Adapt a model trained on a much larger sound corpus | May generalize better from our small subset | Most expect higher-rate audio; resampling cannot restore discarded frequencies; Pi/Hailo deployment must be checked |
| One-dimensional waveform model | Learn filters directly from samples | Feature extraction can adapt to the task | Usually needs more examples or pretraining to be competitive |

Before choosing a replacement, make a small **target-room test set** with time-marked examples of all five classes, mixtures, and quiet background. Keep the same train/validation/test splits while comparing one change at a time. More relevant and accurately labeled appliance examples may improve the result more than a more elaborate network. The selected model remains the baseline so each candidate can be compared fairly.

For each batch, the model emits five raw values called logits. `BCEWithLogitsLoss` compares each logit with its corresponding 0/1 target and supplies a training error. Positive examples get extra weight because most clips are negative for any particular class; the per-class ratio is clipped to 1–15. AdamW adjusts the weights to reduce that error. At inference time a sigmoid converts each logit to a 0–1 score independently. This is why a voice and music score can both be high.

The selected run used batch size **32**, learning rate **0.001**, seed **7**, and up to **25 epochs**. An epoch means one pass over the training rows. Every training window gets a random gain from 0.6 to 1.4. With probability **0.3**, training mixes in a different-label clip at balanced loudness and combines their labels. It preferentially pairs voice and music when applicable. This teaches the model to represent overlapping sounds. The selected run used no synthetic silence training examples; the attempted quiet-window candidate performed worse on held-out clips.

After each epoch, validation scores are converted to detections with a per-class threshold. The training code searches thresholds from 0.1 to 0.9 in 0.05 increments to maximize validation F1 for each class. It saves the weights and thresholds from the epoch with the best mean validation F1, and stops after six epochs without improvement. The selected checkpoint came from **epoch 18**, with validation mean F1 about **0.573**. Saved thresholds are approximately voices **0.5**, machines **0.9**, collisions **0.8**, steps **0.7**, and music **0.75**. The JSON interface emits the raw sigmoid scores; thresholds are used for evaluation and can later be used for application decisions.

## 4. Check on held-out clips

The test split was evaluated after model and threshold selection. Precision means the share of predicted detections that matched a clip label. Recall means the share of labeled clips that were detected. F1 balances those two; 1 is perfect and 0 is worst.

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Voices | 0.647 | 0.489 | 0.557 |
| Machines | 0.536 | 0.357 | 0.429 |
| Collisions | 0.470 | 0.705 | 0.564 |
| Steps | 0.386 | 0.550 | 0.454 |
| Music | 0.647 | 0.524 | 0.579 |

Mean test F1 is **0.516**. This is a first working model with meaningful errors, particularly for machines and steps. The results use one loud window per public clip against whole-clip tags, so they are *proxy window metrics*, not a measured detection rate in your room. On 20 equal-loudness synthetic voice/music test mixtures, both scores passed their thresholds in 7 cases; that small check motivated the selected mixed model but does not estimate room performance. Dryers and washing machines have not been individually validated.

## 5. Run inference

For a live stream, the adapter collects about one second of decoded audio, then advances **2,016 samples** per update, roughly **0.5 seconds**. It computes the same spectrogram, runs the CNN, applies sigmoid, and returns a five-key JSON object. An RMS below **0.0005** returns five zeros to suppress digital silence and very faint noise. The real Thingy stream should bypass the *simulation* step because its firmware already applies that processing; `predict.py live` accepts either decoded samples or compressed codes.

The browser microphone test runs this checkpoint on **CPU** in FastAPI. Docker Compose packages that serving app; it does not train the model. The browser audio is separately shaped to resemble the Thingy stream and sent through a WebSocket. The current `.pt` checkpoint is **not** a Hailo-10H `.hef`; NPU conversion and Raspberry Pi timing remain separate tasks.

## Repeat training or expand the dataset

From this directory, with the Python environment installed and the current manifest clips present:

```bash
python dataset.py status
OMP_NUM_THREADS=8 ./.venv/bin/python train.py \
  --epochs 25 --batch-size 32 --mix-probability 0.3 \
  --silence-probability 0 --output artifacts/candidate.pt
```

This writes `artifacts/candidate.pt` and `artifacts/candidate.metrics.json`, preserving the selected checkpoint while you compare results. On the RTX 2060 Super computer, install CUDA-enabled PyTorch and check `torch.cuda.is_available()` first; `train.py` selects CUDA automatically when available. The browser app can then serve a new selected checkpoint after its file is replaced and the Compose container is recreated.

To grow the public subset, increase `dataset.py select`'s `--train-per-class`, `--val-per-class`, and `--test-per-class` limits, download the newly selected clips, and rerun training. **Back up `data/manifest.jsonl` before reselecting:** selection writes a new manifest and may change which rows are included, even with the same seed. Preserve split separation and compare against untouched target-room recordings. Higher quality labels for real household appliances and short events are likely to help more than simply adding arbitrary clips.

The next meaningful evaluation is a small set of target-room recordings from the Thingy mic: examples of each class, overlapping voice and music, and quiet background. Recordings can remain test-only while keeping the first model based on public training data. Judge scores and latency on that set before treating the heatmap colors as reliable.
