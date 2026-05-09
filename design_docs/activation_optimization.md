## Research note: activation-guided whole-model morphing

### Problem

Current merge search is hitting a ceiling: it can reduce some noise, but subject noise, blurriness, and blotchy textures remain hard to remove. Your stated target is still: keep model A’s knowledge/quality while gaining B-like RF behavior and EQ-VAE compatibility. 

Merge-only findings suggest this is not a single-layer problem. Some parameters behave like global rails: `time_embed_*` should stay high, `to_out_0_*` should stay zero, `skip_connection_rank_ratio` prefers mid values, and `norm_res` / `norm_attn` / `emb_layers` shape texture/noise/detail. 

### Core hypothesis

The bad merges are not merely “not RF enough.”

They may be creating **activation states that neither parent model naturally occupies**.

So the next thing to study is:

```text
A = vpred source
B = RF target
M = merge candidate

Does M behave like a meaningful interpolation of A/B internally,
or does it create abnormal activation / CFG / timestep statistics?
```

This gives you a behavior objective before generating full images.

---

## 1. Build a fixed probe set

Use no full sampling at first.

For each model, run identical:

```text
prompts × negative prompts × latents × timesteps
```

Suggested small set:

```text
prompts: 32–128
latents per prompt: 2–4
timesteps: 8–16, SNR-spaced if possible
models: A, B, best merge, bad merge, several historical high/low trials
```

Include prompt categories:

```text
simple subject
multi-subject
dark background / blackness tests
high-detail character
flat-color / low-texture
problem prompts that produce blotches
```

Save:

```text
model output prediction
CFG delta = cond_pred - uncond_pred
block input/output activations
residual update norm
attention output norm
norm layer statistics
timestep embedding activations
```

Do not start with every tensor. Start with block-level hooks because your own closeout says block-level conclusions are more reliable than attn/ffn/proj splits. 

---

## 2. Activation envelope test

For each probe and block, compute the parent envelope:

```text
low  = min(stat(A), stat(B))
high = max(stat(A), stat(B))
```

Then check whether candidate merge `M` is inside or outside that envelope.

Stats to compute:

```text
activation RMS
activation mean/std
residual update RMS
output/input norm ratio
CFG delta RMS
cosine similarity to A output
cosine similarity to B output
```

Useful metric:

```text
OOD_rate(block, timestep) =
  fraction of probes where M is outside [A, B] envelope
```

Why this matters: if blotchy merges have high OOD rate in `OUT03/OUT04`, `norm_res`, or `emb_layers`, that gives you a concrete non-image proxy for the artifact wall. Your findings already point to `OUT03_1`, `OUT04_1`, `emb_layers`, `norm_res`, and `skip_connection_rank_ratio` as high-value axes. 

Expected useful result:

```text
Good merge:
  moves toward B on timestep/CFG behavior
  stays inside A/B activation envelope

Bad merge:
  may score RF-like in some places
  but creates activation spikes or CFG weirdness in texture/detail blocks
```

---

## 3. RF-behavior signature test

Define “RF-like” behavior from B, not from assumptions about layers.

Compare A vs B across probes:

```text
D_rf(block, t) = distance(behavior_A, behavior_B)
```

Then compare candidate M:

```text
progress_to_B = 
  distance(A, B) - distance(M, B)
```

Do this for:

```text
model output prediction
CFG delta curve over timestep
timestep embedding activations
block residual updates
activation norm profile
```

You are looking for the RF signature as a **trajectory over timesteps**, not a static layer replacement.

Very concrete plots:

```text
x-axis: timestep / sigma / logSNR
y-axis: CFG delta RMS

lines:
A
B
best merge
bad merge
```

And:

```text
x-axis: block depth
y-axis: activation OOD rate or progress-to-B

separate curves per timestep band:
early/noisy
middle
late/clean
```

If `time_embed_*` is truly tied to RF/global scheduling stability, it should show up here as different timestep response curves, not just as an Optuna importance item. Your docs explicitly mark `time_embed` as RF/global stability and a lock-high rail. 

---

## 4. Compare good vs bad merge candidates

Use historical trials, not only A/B.

Group candidates:

```text
parent A
parent B
best merges > 6.8
decent merges 6.0–6.8
bad/artifact merges < 5
known hard failures excluded
```

Important: exclude scorer/runtime failures because your logs include cases where `0.0` scores came from exceptions, not real bad images. 

Then find behavior features that separate groups:

```text
feature: OUT04 activation OOD rate at mid timesteps
feature: CFG delta overshoot vs both parents
feature: norm_res variance
feature: timestep embedding cosine to B
feature: skip-connection output/input ratio
```

This is basically “importance analysis,” but over activations instead of merge parameters.

---

## 5. Candidate whole-model transforms to try

After you have behavior metrics, try transforms that operate across the model, but are scored by probes.

### A. Envelope-constrained merge

Instead of only:

```text
W = A + α(B - A)
```

use candidate α values, then reject or penalize if activations leave the A/B envelope.

Objective:

```text
score =
  RF_progress
  - λ * activation_OOD
  - β * CFG_overshoot
```

This is still merge-like, but the selection pressure is now behavioral.

### B. Delta damping by behavior risk

For each tensor/block group:

```text
Δ = B - A
```

Keep the delta, but damp groups whose activation probes produce out-of-envelope behavior.

```text
W_candidate = A + s_group * Δ_group
```

Where `s_group` is not chosen only from image scores, but from behavior safety:

```text
large s if RF_progress improves and OOD stays low
small s if RF_progress improves but OOD spikes
zero/negative if it creates parent-unnatural behavior
```

This connects directly to your existing finding that `skip_connection_rank_ratio`, `in_layers_rank_ratio`, and several support controls prefer partial values, not max replacement. 

### C. Spectral delta filtering

For each weight delta:

```text
ΔW = W_B - W_A
```

Try:

```text
low-rank keep
high-rank keep
singular value clipping
outlier singular value removal
delta RMS normalization
```

Then run probes. The question is:

```text
Which spectral part of the RF delta moves behavior toward B
without creating activation OOD?
```

This is more whole-model than layer targeting.

### D. Norm-preserving morph

Bad artifacts may correlate with activation norm explosions or weird variance. Try transforms that preserve A’s per-tensor norm while moving direction toward B:

```text
W = normalize_like_A(A + αΔ)
```

Variants:

```text
per-tensor RMS preserve
per-channel RMS preserve
per-block RMS preserve
norm layers excluded/included separately
```

Then check whether artifacts correlate with reduced activation OOD.

---

## 6. First concrete implementation target

Add a command like:

```bash
sd-optim probe-behavior \
  --models A.safetensors B.safetensors best.safetensors bad.safetensors \
  --prompts probe_prompts.txt \
  --timesteps logsnr:12 \
  --latents-per-prompt 2 \
  --hooks block,residual,cfg,time_embed,norm \
  --out behavior_report/
```

Report tables:

```text
RF progress by block/timestep
activation OOD rate by block/timestep
CFG curve distance
top abnormal blocks in bad merges
top stable RF-like blocks in good merges
```

Then add:

```bash
sd-optim morph-search \
  --source A.safetensors \
  --target B.safetensors \
  --objective rf_progress_minus_ood \
  --transforms block_scale,spectral_filter,norm_preserve \
  --out candidates/
```

Only after that, generate images for the top few candidates.

---

## What I would look for first

The highest-value first question:

```text
Do blotchy/bad merges produce activation statistics outside both parent models?
```

Especially in:

```text
OUT03_1
OUT04_1
OUT05_1 support
norm_res
norm_attn
emb_layers
skip connections
time_embed response
```

If yes, you have a new guiding principle:

> Good RF morphs should move toward B while staying inside the A/B behavior envelope.

That would explain why merge-only scoring plateaus: image scorers can find better recipes, but they do not know whether the internal model state has become parent-unnatural until the artifact appears.
