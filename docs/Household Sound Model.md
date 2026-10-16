---
title: Household sound model — signal, CNN, and training
aliases:
  - Household sound model walkthrough
tags:
  - audio
  - machine-learning
  - raspberry-pi
---

# Household sound model: from microphone to five scores

> [!summary] What this note describes
> The **selected CPU PyTorch checkpoint** at `../artifacts/household_sound.pt`, trained on a public FSD50K subset. Its output is five independent scores: voices, machines, collisions, steps, and music. This is the current implementation, not a Hailo NPU model. The companion [[TRAINING_EXPLAINED]] note and [run report](../RUN_REPORT.md) contain operational details and measured results.

![[assets/model_pipeline.png]]

## 1. The question the model answers

For a roughly one-second sound window, we want a vector

$$
\mathbf y = [y_{\text{voices}}, y_{\text{machines}}, y_{\text{collisions}}, y_{\text{steps}}, y_{\text{music}}] \in \{0,1\}^{5}.
$$

Each entry says whether that sound is audible. For speech over music, a valid target is $[1,0,0,0,1]$. It is a **multi-label** problem: multiple entries may be 1. The model returns five continuous scores, not a single class name. A score is a sigmoid output, **not** a calibrated probability that a source exists in the room.

The model does not identify a sound-producing object directly. It compares a new audio pattern with statistical patterns learned from labeled recordings. This matters for machines: the current training tags cover fans, printers, microwaves, drills, and power tools, but dryer and washing-machine performance has not been separately measured.

## 2. The data and the microphone constraint

The [FSD50K dataset][fsd50k] supplied **1,665 selected public clips**. We mapped its source tags into our five targets and selected some rain, wind, water, animal, and insect clips with none of those targets. The selected manifest has **1,194 training**, **238 validation**, and **233 test** clips. Source IDs, tags, licenses, and splits are stored in [`data/manifest.jsonl`](../data/manifest.jsonl). The original FSD50K evaluation partition supplies the test split.

The actual Nordic Thingy:53 stream delivers approximately $4{,}031.25$ decoded samples per second after its filter, 4:1 decimation, and 8-bit logarithmic compression. Training clips begin at other rates, so [`audio_model.py`](../audio_model.py) simulates that path: resample to $16{,}125$ samples/s, apply the firmware-derived FIR, decimate by four, high-pass, quantize and expand. This narrows the frequency range to roughly $0$–$2{,}016$ Hz. It is an approximation of the device, not a recording from the target room.

One model window contains $N=4{,}031$ samples, so its duration is

$$
T = \frac{N}{f_s} = \frac{4031}{4031.25}\approx 0.99994\ \text{s}.
$$

Live inference advances $H=2{,}016$ samples between windows, giving an update about every $H/f_s\approx 0.5001$ s after the first window. Overlap helps a short event appear in at least one full window.

FSD50K labels describe whole clips, not exact one-second intervals. During training we choose randomly among the **three loudest** candidate windows per clip; validation and test use the loudest one. This avoids some silent windows but can still attach a whole-clip label to a second in which that event is absent.

## 3. Why the waveform becomes a spectrogram

A waveform is a sequence $x[n]$ of pressure-related samples. A steady fan, a footstep, and a knock can all look like complicated oscillations in that sequence. The short-time Fourier transform (STFT) exposes how their frequency content changes over time:

$$
Z[k,m] = \frac{1}{\sum_{n=0}^{255}w[n]}
\sum_{n=0}^{255}x[n+80m]w[n]e^{-j2\pi kn/256}.
$$

Here $m$ is a time frame, $k$ is a frequency bin, and $w[n]$ is the window used by SciPy's STFT. We use a **256-sample** analysis frame (about $63.5$ ms) and move it by **80 samples** (about $19.8$ ms). A real-valued 256-point FFT gives $256/2+1=129$ nonnegative-frequency bins, spaced by about $f_s/256=15.75$ Hz. Across 4,031 samples there are

$$
1+\left\lfloor\frac{4031-256}{80}\right\rfloor=48
$$

complete frames. The model's input feature is therefore $1\times129\times48$: one channel, 129 frequencies, 48 time positions.

The complex STFT includes phase. We retain its magnitude and compress it:

$$
F[k,m] = \log\bigl(1+1000\lvert Z[k,m]\rvert\bigr).
$$

The logarithm prevents strong components from overwhelming weaker ones. `1000` is a fixed scaling choice. We do **not** apply mel filters, MFCCs, or per-clip normalization in this model.

![[assets/spectrogram_examples.png]]

*These plots are real outputs of the model's preprocessing code, showing the loudest one-second training window for each FSD50K clip. Fan energy persists; the wooden knocks form brief vertical bands; the footsteps show uneven bursts. Each panel's display color range is separately capped at its 99th percentile, so colors must not be compared as absolute loudness across panels. The model receives the uncapped numerical features. Source clips: [fan 104017](https://freesound.org/s/104017/) and [knocks 104001](https://freesound.org/s/104001/) by RutgerMuller, [footsteps 119913](https://freesound.org/s/119913/) by ftpalad; their manifest entries report CC0 licenses.*

This resembles image classification, as the [Hugging Face audio course][hf-classification] explains. The axes have different meanings, however. Moving a pattern later in **time** preserves the source; moving it in **frequency** changes pitch and can change what the source sounds like. Our two-dimensional CNN is a compact baseline, not a claim that spectrograms behave exactly like photographs.

### Why 2D spectrograms instead of 1D raw waveforms or Transformers?

When selecting an audio representation for edge AI, three major paradigms exist:

| Representation | Processing Mechanism | Edge Suitability | Rationale for Our Choice |
| :--- | :--- | :--- | :--- |
| **1D Raw Waveform** *(e.g. WaveNet, SampleCNN)* | 1D time-domain convolutions on raw samples | ❌ High parameter count, phase-sensitive | Oscillates at kHz. A 1-sample shift ($0.25\text{ ms}$) flips signal polarity from $+1$ to $-1$. A network must dedicate immense parameter capacity just to learn phase-invariant Fourier filters from scratch. |
| **Transformer / AST** *(Self-Attention)* | Patch extraction on spectrogram + Multi-Head Attention | ❌ High memory & quadratic $O(N^2)$ complexity | Attention across time-frequency patches requires millions of parameters and heavy memory bandwidth, incompatible with sub-4 ms latency on edge CPUs and low-power NPUs. |
| **2D Spectrogram CNN** *(SoundCNN)* | Deterministic STFT + 2D Spatial Convolutions | ✅ **Optimal for Edge IoT** | STFT mathematically extracts energy distribution in $O(N \log N)$ compute. Discarding phase provides natural phase-invariance, while $3\times 3$ kernels efficiently detect acoustic geometry (lines, bursts, bands). |

### The time-frequency resolution trade-off (Heisenberg Gabor limit)

The choice of $N_{\text{FFT}}=256$ and hop $80$ is a deliberate physical compromise:
- **Frequency resolution**: $f_s / N_{\text{FFT}} = 4{,}031.25 / 256 \approx \mathbf{15.75\text{ Hz}}$ per bin. This is fine enough to separate low-frequency motor hums and power harmonics ($50\text{ Hz}$, $100\text{ Hz}$, $150\text{ Hz}$) without blurring them together.
- **Time resolution**: $\text{hop} / f_s = 80 / 4{,}031.25 \approx \mathbf{19.8\text{ ms}}$ per frame (~50 frames/second). This is fast enough to localize transient impact spikes (door knocks, dropped objects, footstep heel strikes) within one or two frames.
- Increasing $N_{\text{FFT}}$ to 512 would double frequency resolution but stretch the time window to $127\text{ ms}$, blurring brief impact transients into diffuse smudges. Conversely, reducing $N_{\text{FFT}}$ to 128 would blur fundamental pitch harmonics.

## 4. One CNN layer, in detail

The first block in [`SoundCNN`](../audio_model.py) is:

```text
Input [batch, 1, 129, 48]
  → Conv2d(1, 16, kernel=3×3, padding=1)
  → BatchNorm2d(16)
  → ReLU
  → MaxPool2d(2)
Output [batch, 16, 64, 24]
```

**Convolution.** Each of 16 filters has a learned $3\times3$ grid of weights and one bias. At frequency $f$ and time $t$, one filter computes a weighted sum of the nearby feature values:

$$
C_c[f,t] = b_c + \sum_{u=-1}^{1}\sum_{v=-1}^{1}K_c[u,v]F[f+u,t+v],
\qquad c\in\{1,\ldots,16\}.
$$

Outside the feature's edges, padding supplies zeros. PyTorch slides these weights over the grid; its `Conv2d` operation uses cross-correlation, so the kernel is not flipped in this equation. The result is **16 new maps, each $129\times48$**. A filter can respond to a local burst, stripe, edge, or another useful pattern; its actual learned weights are determined by training.

For a deliberately simple example, imagine one $3\times3$ patch and one *illustrative* kernel:

$$
P=\begin{bmatrix}0&0&1\\0&2&1\\0&1&0\end{bmatrix},\quad
K=\begin{bmatrix}-1&0&1\\-1&0&1\\-1&0&1\end{bmatrix},\quad b=0.
$$

Their elementwise products add to $1+1+0=2$. This example kernel responds to stronger values on the right (later time) than on the left. **These are invented numbers for explanation, not weights read from the checkpoint.** One learned filter has $3\cdot3+1=10$ parameters, so the 16 convolutions have $160$.

![[assets/first_convolution.png]]

**Batch normalization.** For each output channel, it transforms the convolution value using a learned scale $\gamma_c$ and offset $\beta_c$:

$$
B_c=\gamma_c\frac{C_c-\mu_c}{\sqrt{\sigma_c^2+\epsilon}}+\beta_c.
$$

During training $\mu_c$ and $\sigma_c^2$ come from the batch; during inference, saved running estimates are used. The 16 scales and 16 offsets add **32 learned parameters**.

**ReLU and max-pool.** ReLU changes each value to $\max(0,B_c)$, allowing nonlinear features. A $2\times2$ max-pool keeps the largest value in each non-overlapping neighborhood. That reduces $129\times48$ to $64\times24$ (the extra frequency row is dropped). The block therefore has **$160+32=192$ learned parameters**. Pooling reduces compute and sensitivity to exact position, but discards detail.

The next blocks repeat the pattern with 32, 64, then 96 channels. Their pooled map sizes are $32\times12$, $16\times6$, and $8\times3$. All four blocks and the linear head total **79,589 learned parameters**.

### Why this specific topology? (Computational balancing and receptive field)

The architecture uses an **inverted pyramid** design common in edge vision and audio backbones:

| Stage | Input Shape | Conv & Pooling Config | Output Shape | Receptive Field | Compute (MACs) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Input** | `1 × 129 × 48` | Spectrogram input | `1 × 129 × 48` | $1 \times 1$ | 0 |
| **Block 1** | `1 × 129 × 48` | Conv $1\to 16$, BN, ReLU, Pool(2) | `16 × 64 × 24` | $4 \times 4$ | $\approx 0.22\text{ M}$ |
| **Block 2** | `16 × 64 × 24` | Conv $16\to 32$, BN, ReLU, Pool(2) | `32 × 32 × 12` | $10 \times 10$ | $\approx 1.77\text{ M}$ |
| **Block 3** | `32 × 32 × 12` | Conv $32\to 64$, BN, ReLU, Pool(2) | `64 × 16 × 6` | $22 \times 22$ | $\approx 1.77\text{ M}$ |
| **Block 4** | `64 × 16 × 6` | Conv $64\to 96$, BN, ReLU, Pool(2) | `96 × 8 × 3` | $46 \times 46$ | $\approx 1.33\text{ M}$ |
| **Head** | `96 × 8 × 3` | AdaptiveAvgPool2d(1) + Linear(96→5)| `5` logits | Full window | $\approx 0.001\text{ M}$ |
| **Total** | — | — | — | — | **~5.1 MFLOPs (79,589 params)** |

1. **Uniform Arithmetic Budget**: Notice that as spatial dimensions are halved by pooling ($64\times 24 \to 32\times 12 \to 16\times 6 \to 8\times 3$), channel depth doubles ($16 \to 32 \to 64 \to 96$). This keeps the computational load per block nearly uniform (~1.5–1.7 MFLOPs), preventing any single layer from bottlenecking the CPU or NPU pipeline.
2. **Receptive Field Growth**: Each $3\times 3$ convolution combined with $2\times 2$ pooling expands the context seen by each neuron. By Block 4, each feature point integrates a $46 \times 46$ patch of the input spectrogram, allowing the network to recognize multi-frame acoustic rhythm (such as repetitive walking cadence or speech syllables).

### Why Global Average Pooling (`AdaptiveAvgPool2d(1)`) instead of Dense Flattening?

In early CNNs (such as AlexNet or VGG), the final convolutional feature maps were flattened into dense fully connected layers. Applying that classical approach here would have been catastrophic:
- Flattening `96 × 8 × 3` yields **2,304 features**.
- Connecting that to a standard 512-neuron dense layer would require:
  $$2{,}304 \times 512 = \mathbf{1{,}179{,}648\text{ parameters!}}$$
  That single dense layer would consume **94% of the entire model's memory**, causing massive memory bandwidth stalls on the Raspberry Pi and severely overfitting on small datasets.
- Instead, **Global Average Pooling (GAP)** computes the mean of each $8\times 3$ map, condensing each channel to a single feature activation. The subsequent classification head is simply `Linear(96, 5)`:
  $$96 \times 5 + 5 = \mathbf{485\text{ parameters!}}$$
- **Acoustic Advantage**: GAP imparts **temporal shift-invariance**. An impact sound (like a knock or dish collision) occurring at $t=0.2\text{ s}$ or $t=0.8\text{ s}$ activates the corresponding convolutional filter equally, yielding consistent detection regardless of when the sound begins inside the 1-second window.

### Edge and NPU deployment considerations

The choice of operations was directly driven by edge deployment constraints (Raspberry Pi Cortex-A76 CPU and Hailo-10H NPU):
- **Zero-Cost BatchNorm Folding**: In evaluation mode, Batch Normalization is a fixed linear transformation:
  $$B_c = \gamma_c \frac{C_c - \mu_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$
  Because it directly follows a convolution ($C_c = W x + b$), it is mathematically folded into the preceding convolution's weights and biases during compilation:
  $$W_{\text{fused}} = W \cdot \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}}, \quad b_{\text{fused}} = (b - \mu_c) \frac{\gamma_c}{\sqrt{\sigma_c^2 + \epsilon}} + \beta_c$$
  This completely eliminates 4 layers of memory reads/writes at runtime with zero loss in numerical accuracy.
- **Hardware-Native ReLU**: $f(x) = \max(0, x)$ requires a single comparison instruction on CPUs and is natively implemented in dedicated hardware logic on the Hailo-10H NPU. It avoids expensive transcendental functions (like GELU or Swish) and creates activation sparsity that improves 8-bit integer post-training quantization (PTQ).

### Multi-label classification with independent Sigmoids

The 96-to-5 outputs are raw *logits* $z_c$, converted to continuous confidence scores by independent sigmoids:

$$
p_c = \sigma(z_c) = \frac{1}{1+e^{-z_c}}.
$$

**Why not Softmax?** Softmax enforces mutual exclusivity ($\sum_{c=1}^5 p_c = 1$), assuming that only one sound can exist at any given moment. In physical household environments, acoustic events are additive: a person can speak while background music plays (`voices` = 0.85, `music` = 0.78), or footsteps can occur while a washing machine hums. Independent sigmoids allow all five classes to be scored simultaneously without artificial competition.

## 5. How the weights were trained

The network starts with random learned weights. For each batch, [`train.py`](../train.py) performs this cycle:

1. Read 32 training rows and choose a window for each. Convert those windows with the exact preprocessing above; processed clips are cached to avoid repeating resampling.
2. Multiply each training window by a random gain between $0.6$ and $1.4$. With probability $0.3$, add a balanced-loudness window from a different class and combine the two label vectors with logical OR. Speech and music are preferentially paired when applicable.
3. Run the batch through the CNN to obtain five logits for every example.
4. Measure error against its five binary labels with **weighted binary cross-entropy**. For class $c$, with target $y_c$, score $p_c$, and positive weight $w_c$, the term is

   $$
   \ell_c=-w_c y_c\log p_c-(1-y_c)\log(1-p_c).
   $$

   The program averages these terms over batch items and classes. It chooses $w_c=\operatorname{clip}((N_{\text{train}}-P_c)/P_c,1,15)$, where $P_c$ is the number of positive training rows for that class. Positive examples therefore matter more when they are rarer.
5. Backpropagation calculates how changing each learned weight would change the error. **AdamW**, with learning rate $0.001$, updates the weights. The process repeats for the next batch.

For intuition, if the correct music label is 1 but the music score is $0.2$, its positive loss is $-w_{\text{music}}\log(0.2)$: relatively large. The gradient tends to move the involved weights toward a higher music logit on similar examples. There is no hand-written “music filter”; the network discovers weights through many such updates.

One full pass over training rows is an **epoch**. The selected run allowed up to 25 epochs, used seed 7, and saved the best checkpoint at **epoch 18**. It stopped after six epochs without a better validation macro F1. The selected run did not use synthetic silence training windows.

## 6. Validation, thresholds, and the final test

After each epoch, the model scores the **validation** split. For each class, the program searches thresholds from $0.1$ through $0.9$ in steps of $0.05$ and picks the threshold with the highest validation $F_1$:

$$
\operatorname{precision}=\frac{TP}{TP+FP},\quad
\operatorname{recall}=\frac{TP}{TP+FN},\quad
F_1=\frac{2\operatorname{precision}\operatorname{recall}}{\operatorname{precision}+\operatorname{recall}}.
$$

The mean of the five validation $F_1$ scores selects the checkpoint. Its thresholds are voices $0.5$, machines $0.9$, collisions $0.8$, steps $0.7$, and music $0.75$. These thresholds are **for turning scores into yes/no decisions**; the JSON heatmap interface emits raw sigmoid scores. A separate RMS gate returns five zeros for an essentially silent window ($\operatorname{RMS}<0.0005$).

Only after model and threshold selection does the code score the held-out test split. Test $F_1$ was **0.557 voices, 0.429 machines, 0.564 collisions, 0.454 steps, and 0.579 music**; mean $F_1$ was **0.516**. These are proxy one-second scores checked against whole-clip tags, not measured reliability in your room. Choosing both epoch and thresholds on the same validation set can also make validation results optimistic; the held-out test reduces that concern but does not solve the room/device mismatch.

> [!important] What to test next
> Record short, time-marked examples from the fixed room using the Thingy mic: each of the five categories, quiet background, and overlapping voice/music. Keep them separate from training initially. Compare errors and latency before changing the feature extraction or CNN. More accurate appliance examples may help more than a larger network.

## 7. Why this particular baseline, and alternatives

The log spectrogram plus small CNN was inexpensive to train and gives a usable first checkpoint. It uses a familiar architecture for a time-frequency array but pools **both axes** four times. Possible experiments include preserving more time positions, using temporal max or attention pooling for short impacts, testing log-mel features, or adapting a pretrained audio encoder. Each change should be measured against the same splits and the target-room set. The linked [Hugging Face lesson][hf-classification] discusses spectrogram-based ConvNets and their difference from ordinary images; its transformer architectures are **background reading**, not models used in this project.

## Bibliography and project references

1. [Hugging Face Audio Course, “Architectures de classification d’audio”][hf-classification] (French). Explains spectrogram-based audio classification and why the frequency axis differs from an image axis.
2. [Fonseca et al., “FSD50K: An Open Dataset of Human-Labeled Sound Events”][fsd50k-paper]. Describes the public dataset used for the selected clips.
3. [FSD50K dataset record][fsd50k]. Dataset files and license context; consult each selected clip's license in the manifest before redistribution.
4. Project code: [`audio_model.py`](../audio_model.py) for signal processing and the CNN; [`train.py`](../train.py) for optimization; [`predict.py`](../predict.py) for inference; [run report](../RUN_REPORT.md) for measured results.

**Figure generation:** Run `python3 docs/generate_figures.py` from the project root. It needs the project `.venv` with the downloaded clips, and a system Python with Matplotlib. The script calls the actual preprocessing code for the spectrograms and writes the two PNG files under `docs/assets/`.

[hf-classification]: https://huggingface.co/learn/audio-course/fr/chapter3/classification
[fsd50k-paper]: https://arxiv.org/abs/2010.00475
[fsd50k]: https://zenodo.org/records/4060432
