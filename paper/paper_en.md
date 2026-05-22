# LoRA Fine-Tuning of a Personally-Created Character for Image Generation: An Empirical Study on a Consumer-Grade GPU

**Author:** Masato Suzuki (Masafy)
**Date:** May 22, 2026
**Environment:** Home GPU PC (NVIDIA GeForce RTX 3060)

---

## Abstract

This report describes LoRA (Low-Rank Adaptation) fine-tuning of Stable Diffusion 1.5 for the purpose of generating novel illustrations of an original character, "Masafee" — a red-panda character created by the author and published as a LINE sticker set. No paid cloud computing resources were used; all training was performed on a single consumer-grade NVIDIA GeForce RTX 3060 (12 GB VRAM) installed in the author's home PC. From an existing set of 16 stickers, 13 were selected by quality criteria as training data. Three experiments were conducted: (1) an initial training run to verify feasibility, (2) a six-condition comparison varying the base model, learning rate, and LoRA rank, and (3) an epoch ablation from 10 to 60 epochs to identify the onset of overfitting. The results show that (a) the choice of base model has the largest impact on output quality — using an illustration-oriented base model resolves problems with background and art style; (b) training performance plateaus at roughly 30 epochs, beyond which there is no improvement and mild overfitting appears; and (c) the upper bound on output quality is governed by the number of training images. The study confirms that a practical character LoRA can be produced entirely on a single consumer-grade GPU.

**Keywords:** image generation, Stable Diffusion, LoRA, fine-tuning, character generation, consumer-grade GPU

---

## 1. Introduction

Text-to-image diffusion models have become widely available, and open models such as Stable Diffusion can now be used by individuals. However, generating a specific original character with a consistent appearance is not feasible with a general-purpose model alone; additional training is required.

LoRA (Low-Rank Adaptation) trains only low-rank difference matrices instead of re-training the entire large model. It allows a specific concept to be added to a model with modest compute and few training images, making the fine-tuning of a personally-owned character realistic.

The motivation of this work is practical. The author created an original red-panda character, "Masafee," and published it as a LINE sticker set of 16 stickers. The starting point is the desire to **generate new poses and scenes of this character without drawing them by hand.**

While paid cloud computing (e.g., Google Colab Pro+) was initially considered, this study instead verifies whether the task can be completed using **only a single consumer-grade GPU** already available at home. The specific questions are:

- **Q1.** Can a character's identity be preserved while generating novel poses, using only ~13 images?
- **Q2.** What factors govern output quality (base model, learning rate, LoRA rank, epoch count)?
- **Q3.** Does increasing the number of training epochs continue to improve quality?

---

## 2. Background

**Stable Diffusion** is a latent diffusion model; this study targets the widely-used SD 1.5 family. **LoRA** trains only a low-rank difference $\Delta W = BA$ of a weight matrix $W$, drastically reducing the number of trainable parameters. Training used `sd-scripts` by kohya-ss, one of the de facto standard implementations for LoRA training.

---

## 3. Method

### 3.1 Computing Environment

| Item | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM, Ampere, compute capability 8.6) |
| CPU / RAM | 20 logical cores / 30 GB |
| OS | Ubuntu 26.04 LTS |
| Training framework | kohya-ss/sd-scripts, PyTorch 2.5.1 + CUDA 12.4 |
| Python | The system Python (3.14) was incompatible with the training tools; Python 3.11 was provisioned separately using the `uv` tool. |

Because the training toolchain could not resolve its dependencies on the OS-default Python 3.14, an isolated Python 3.11 environment was built with the lightweight Python manager `uv`. This is a practical consideration when building a training environment on a bleeding-edge distribution.

### 3.2 Training Data

The training data originated from the author-created LINE sticker set "Masafee" (16 stickers). As shown in Figure 1, the character is based on a red panda and has consistent features: goggles, a crown-patterned bandana, and a ringed tail.

Of the 16 stickers, 3 were excluded and **13** were used for training, by the following criteria:

- Severely low resolution: 1 image
- Rough-sketch style inconsistent with the rest: 2 images

All 13 selected images are 1000×1000 px RGB: 8 face close-ups, 3 full-body, 1 upper-body, and 1 distinct scene.

![Figure 1: 13 training images](../学習データ_13枚.png)
**Figure 1.** The 13 Masafee stickers used for training.

### 3.3 Caption Design

Each image was given a caption referenced during training. The policy was as follows:

- A unique trigger word `masafee` was fixed at the start of every caption.
- **Attributes intended to be variable** (pose, expression, background, presence of text) were written explicitly in captions, so they would be learned as independent, controllable concepts.
- **Invariant character features** (goggles, bandana, etc.) were deliberately *not* captioned, so that they would bind to the trigger word `masafee` itself.

Because text such as "OK!" is baked into the sticker images, rather than cropping it out, the captions noted "with text" so the model would learn to separate text from the character.

### 3.4 Training Configuration

The optimizer was 8-bit AdamW, mixed precision was bfloat16, and LoRA was trained including the text encoder. The attention mechanism used PyTorch's native SDPA, avoiding a dependency on additional libraries (xformers).

---

## 4. Experiments

Three experiments were conducted.

### Experiment 1: Initial Training (Feasibility)

Initial training used standard SD 1.5 (`v1-5-pruned-emaonly`) as the base, 512 px resolution, LoRA rank 32 / alpha 16, learning rate 1e-4, batch size 2, for 16 epochs (1040 steps).

### Experiment 2: Condition Sweep (Factor Comparison)

To identify the factors governing output quality, the following six conditions were trained and compared automatically. Resolution was 768 px, with 30 epochs each (weights saved at epochs 10/20/30).

- Base model: standard SD 1.5 / Counterfeit-V3.0 (an illustration-oriented model)
- Learning rate: 1e-4 / 2e-4
- LoRA rank: 16 (alpha 8) / 32 (alpha 16)

For each checkpoint, images were generated with five evaluation prompts absent from the training data (waving, jumping, holding coffee, angry face, sleeping), and comparison grids were produced.

### Experiment 3: Epoch Ablation

For the best condition from Experiment 2, training was **extended to 60 epochs**, and the outputs at epochs 30, 40, 50, and 60 were compared. With a fixed random seed, results up to epoch 30 reproduce Experiment 2 identically, with epochs 40–60 added.

---

## 5. Results

### 5.1 Experiment 1: Initial Training Succeeds

Initial training took about 9 minutes on the RTX 3060, using about 3 GB of VRAM. In generation tests, the character's identity (goggles, bandana, facial markings) was reproduced well, and poses absent from the training data (jumping, sleeping, astronaut, etc.) could be generated. **Q1 is confirmed affirmatively.**

Two weaknesses were observed, however: (i) backgrounds turned dark even when "white background" was specified in the caption, and (ii) outputs had stronger shading than the flat art style of the original stickers.

### 5.2 Experiment 2: The Base Model Matters Most

The six-condition comparison gave a clear result (Figure 2).

![Figure 2a: SD1.5 base](../スイープ比較/sd15_lr1_d32.png)
![Figure 2b: Counterfeit base](../スイープ比較/cf_lr1_d32.png)
**Figure 2.** Representative sweep comparison. Top: standard SD 1.5 base. Bottom: Counterfeit base. Rows are epochs (10/20/30); columns are evaluation prompts.

- The three **standard-SD1.5-base** conditions all darkened backgrounds and applied strong shading (the same weaknesses as Experiment 1).
- The three **Counterfeit-base** conditions produced clean white backgrounds and flat coloring close to the original sticker art style.

That is, the two weaknesses observed in Experiment 1 were resolved **not by tuning the learning rate or LoRA rank, but by switching the base model to an illustration-oriented one.** Differences due to learning rate and LoRA rank were small compared to the difference due to the base model. **For Q2, we conclude that among the factors examined, the base model choice is dominant.**

The best condition was `Counterfeit + learning rate 1e-4 + LoRA rank 32` (hereafter cf_lr1_d32).

### 5.3 Experiment 3: Epochs Plateau at About 30

Figure 3 shows the best condition extended to 60 epochs.

![Figure 3: Comparison of epochs 30-60](../スイープ比較/延長テスト_epoch30-60.png)
**Figure 3.** cf_lr1_d32 compared at epochs 30/40/50/60. Differences between rows are slight.

The outputs at epochs 30, 40, 50, and 60 were nearly indistinguishable; **performance had already plateaued by epoch 30.** At epochs 50–60, saturation in fact increased, showing slight signs of overfitting ("baking"). **For Q3, we conclude that, for this data and configuration, increasing the epoch count beyond about 30 does not improve quality.**

### 5.4 Generation with the Final Model

Using the best model (cf_lr1_d32, epoch 30), 12 scenes absent from the training data (dancing, eating ramen, reading, playing guitar, cooking, running in the rain, etc.) were generated (Figure 4). All preserved the character's identity and the flat art style, at a quality sufficient for practical use.

![Figure 4: Novel poses generated by the final model](../ショーケース_一覧.png)
**Figure 4.** Twelve novel poses generated by the best model.

For "running in the rain," the background was initially dark and ominous. This was because the prompt did not specify background brightness; specifying "bright, soft colors, light blue background" and adding "dark, scary" to the negative prompt resolved it. This is a practical lesson that scene-dependent attributes should be specified explicitly.

---

## 6. Discussion

**Dominance of the base model.** The main finding of this study is that output quality strongly depends on the choice of base model. When the target character's art style is a flat illustration, a photorealism-oriented base model conflicts with that style and harms backgrounds and shading. Choosing a base model that matches the target art style takes priority over hyperparameter tuning.

**Overfitting and epoch count.** With as few as 13 images, training saturates at about 30 epochs. Training beyond that does not improve quality and increases the risk of overfitting (memorizing training images, losing the ability to generate novel poses, fixing backgrounds and colors). The intuition that "longer training is better" is incorrect; an appropriate epoch count exists, depending on data size.

**Data size sets the quality ceiling.** Since the epoch count plateaus, the main remaining lever for further quality gains is adding training images. The figure of 13 images itself defines the achievable quality ceiling.

**Sufficiency of consumer-grade GPUs.** Each training run took 9–40 minutes on the RTX 3060, using 3–8 GB of VRAM. For creating an SD1.5 character LoRA, paid cloud computing is not required; a single consumer-grade GPU suffices, including iterative experimentation.

---

## 7. Limitations and Future Work

- **Subjectivity of evaluation.** Evaluation here was by visual inspection of comparison grids, without quantitative metrics (CLIP similarity, FID, etc.).
- **Data scale.** Training used 13 images of a single character, limiting generalizability.
- **Baked-in text.** Text in the training images was handled by caption description rather than cropping; the effect of removal is untested.
- **Sweep variance.** Each condition was evaluated with a single seed, so seed-induced variance is not captured.

Future work includes improving quality by adding training images, introducing quantitative evaluation metrics, and validation with newer base models such as SDXL.

---

## 8. Conclusion

Using only a consumer-grade GPU (RTX 3060), we performed LoRA fine-tuning for image generation of the author's original character "Masafee." We confirmed that, from 13 training images, novel poses can be generated while preserving the character's identity (Q1). Through a six-condition comparison, we showed that output quality depends most strongly on the choice of base model (Q2). Furthermore, through an epoch ablation from 10 to 60, we demonstrated that performance plateaus at about 30 epochs and that further training does not improve quality (Q3). This study shows that fine-tuning a personally-owned character can be completed practically on a single consumer-grade GPU, without paid cloud resources.

---

## Appendix A: Key Hyperparameters (Best Condition, cf_lr1_d32)

| Parameter | Value |
|---|---|
| Base model | Counterfeit-V3.0 (SD1.5-family illustration model) |
| Resolution | 768 × 768 |
| LoRA rank / alpha | 32 / 16 |
| Learning rate | 1e-4 (cosine scheduler) |
| Optimizer | 8-bit AdamW |
| Batch size | 2 |
| Epochs | 30 (recommended); ~1950 steps total |
| Mixed precision | bfloat16 |
| Data augmentation | Horizontal flip |
| Attention | SDPA |

## Appendix B: Reproducibility

On the GPU PC, the full training environment is kept in `~/masafee-lora/`, the sweep results in `~/masafee-lora/sweep/`, and the epoch-extension experiment in `~/masafee-lora/sweep/ext60/`.
