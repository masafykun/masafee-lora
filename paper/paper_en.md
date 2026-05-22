# LoRA Fine-Tuning of a Personally-Created Character for Image Generation
## An Empirical Study under Few-Shot Conditions on a Consumer-Grade GPU

**Author:** Masato Suzuki (Masafy)
**Date:** May 22, 2026

---

## Abstract

This report describes fine-tuning of the latent diffusion model Stable Diffusion 1.5 via Low-Rank Adaptation (LoRA), for the purpose of generating novel illustrations of an original character, "Masafee" — a red-panda character created by the author and published as a messaging-app sticker set. No paid cloud computing resources were used; all training was performed on a single consumer-grade NVIDIA GeForce RTX 3060 (12 GB VRAM). From an existing set of 16 stickers, 13 were selected by quality criteria as the training set. Three experiments were conducted: (1) an initial training run to verify feasibility, (2) a six-condition comparison varying the base model, learning rate, and LoRA rank, and (3) an ablation over training epochs from 10 to 60. We show experimentally that (a) the choice of base model has the largest impact on output quality; (b) training performance reaches a plateau at roughly 30 epochs; and (c) the effective upper bound on output quality is governed by the number of training images.

**Keywords:** image generation, latent diffusion model, Stable Diffusion, low-rank adaptation, LoRA, fine-tuning, character generation, consumer-grade GPU

---

## 1. Introduction

Text-to-image diffusion models have advanced rapidly, and open-weight models such as Stable Diffusion are now usable at the individual level. Pre-trained on vast and diverse image corpora, they possess general-purpose generation ability. However, consistently generating a *specific* original character not present in the pre-training distribution, while preserving its identity, is fundamentally difficult with a general-purpose model alone: the corresponding mode is absent, or of extremely low density, in the pre-training distribution. Additional fine-tuning is therefore required.

Naive full-parameter fine-tuning faces two problems: (i) the cost of updating billions of parameters, and (ii) catastrophic forgetting and overfitting under scarce data. Low-Rank Adaptation (LoRA) mitigates both by freezing the pre-trained weights and learning only low-rank difference matrices.

The motivation is practical. The author created an original red-panda character, "Masafee," published as a sticker set (16 stickers). The starting point is to generate unrecorded poses and scenes without additional hand-drawing. While paid cloud computing was initially considered, this study verifies whether the workflow can be completed on a single consumer-grade GPU. The research questions are:

- **Q1.** Can novel poses be generated with preserved identity from ~13 few-shot reference images?
- **Q2.** What factor is rate-limiting for output quality (base model, learning rate, LoRA rank, or epoch count)?
- **Q3.** Does increasing the number of training epochs improve quality monotonically?

## 2. Related Work

This study is situated at the intersection of four research lineages: diffusion models, latent diffusion models, low-rank adaptation, and subject-driven generation.

**Diffusion models.** A practical formulation of diffusion probabilistic models was established by the DDPM (Denoising Diffusion Probabilistic Models) of Ho et al. [1], which fixes the forward process as Gaussian noising and approximates the reverse process with a noise-predicting network. The noise-prediction loss in §3.1 builds on this formulation.

**Latent diffusion models.** Rombach et al. [2] proposed the latent diffusion model (LDM), which runs the diffusion process in the latent space of a variational autoencoder rather than in pixel space, greatly improving the computational efficiency of high-resolution image synthesis. The Stable Diffusion family targeted here is built on this LDM.

**Low-rank adaptation.** LoRA was proposed by Hu et al. [3], originally as an efficient fine-tuning method for large language models. The principle of freezing pre-trained weights and learning only a low-rank difference was subsequently extended to Transformers in general and to the U-Net of diffusion models. This study applies LoRA to Stable Diffusion.

**Subject-driven generation.** The task of binding a specific subject to a model from a few reference images is known as subject-driven generation. The DreamBooth of Ruiz et al. [4] demonstrated a method to bind a specific subject to a text-to-image diffusion model using a unique identifier and few images. The concept binding via the trigger word used in this study belongs to this DreamBooth lineage; in particular, this study corresponds to realizing DreamBooth-style subject binding via LoRA rather than full fine-tuning (colloquially "DreamBooth-LoRA").

## 3. Preliminaries

### 3.1 Diffusion Probabilistic Models

A diffusion model [1] consists of a *forward process* that gradually adds Gaussian noise and a *reverse process* that traces it back. For data $x_0\sim q(x_0)$, the forward process is a Markov chain with variance schedule $\{\beta_t\}_{t=1}^{T}$:

$$q(x_t\mid x_{t-1}) = \mathcal{N}\big(x_t;\,\sqrt{1-\beta_t}\,x_{t-1},\,\beta_t\mathbf{I}\big).$$

With $\alpha_t:=1-\beta_t$ and $\bar\alpha_t:=\prod_{s=1}^{t}\alpha_s$, the state at any step $t$ has the closed form

$$x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1-\bar\alpha_t}\,\epsilon,\qquad \epsilon\sim\mathcal{N}(0,\mathbf{I}).$$

The reverse process is represented by a noise-predicting network $\epsilon_\theta$, and training minimizes the noise-prediction loss

$$\mathcal{L}(\theta)=\mathbb{E}_{x_0,\epsilon,t}\big[\lVert\epsilon-\epsilon_\theta(x_t,t,c)\rVert_2^2\big],$$

where $c$ is the conditioning variable (here, a text embedding).

### 3.2 Latent Diffusion Models and Architecture

The latent diffusion model (LDM) [2] adopted by Stable Diffusion compresses an image into a low-dimensional latent $z_0=\mathcal{E}(x_0)$ via a VAE encoder $\mathcal{E}$, and runs diffusion in latent space; the decoder $\mathcal{D}$ restores pixels. The noise predictor $\epsilon_\theta$ uses a U-Net, into which the text condition $c$ is injected via *cross-attention*. The condition $c$ is the embedding sequence from a text encoder (the CLIP text tower).

### 3.3 Conditional Generation and Classifier-Free Guidance

At inference, classifier-free guidance (CFG) combines conditional and unconditional predictions with guidance scale $s\ge1$:

$$\tilde\epsilon_\theta=\epsilon_\theta(x_t,t,\varnothing)+s\big(\epsilon_\theta(x_t,t,c)-\epsilon_\theta(x_t,t,\varnothing)\big).$$

We set $s\approx7$.

### 3.4 Low-Rank Adaptation (LoRA)

Let a pre-trained weight matrix be $W_0\in\mathbb{R}^{d\times k}$. LoRA [3] freezes it and writes the adapted weight as

$$W' = W_0 + \Delta W,\qquad \Delta W=\frac{\alpha}{r}\,B A,$$

with $B\in\mathbb{R}^{d\times r}$, $A\in\mathbb{R}^{r\times k}$, rank $r\ll\min(d,k)$. The scalar $\alpha$ scales the adaptation ($\propto\alpha/r$). Only $A,B$ are trained ($r(d+k)$ parameters), far fewer than $dk$. Gaussian-initializing $A$ and zero-initializing $B$ guarantees $\Delta W=0$ at the start. We call $r$ the "LoRA rank" and $\alpha$ "alpha."

### 3.5 Terminology

| Term | Description |
|---|---|
| EMA | Exponential moving average; smoothed weights, stored by `emaonly`. |
| Bucketing | Grouping images of differing aspect ratios into resolution buckets. |
| Gradient checkpointing | Recomputing activations to reduce VRAM. |
| Mixed precision (bf16) | Using low precision for speed and memory. |
| SDPA | PyTorch's native fast attention kernel. |
| Trigger word | A unique token bound to the concept. |

## 4. Problem Formulation

Let the target character be $\mathcal{C}$ and its reference set $\mathcal{X}=\{x^{(1)},\dots,x^{(N)}\}$ ($N=13$). Each image has a caption $c^{(i)}$ containing a trigger word $\tau$ (`masafee`). The goal is to learn a LoRA difference $\Delta\theta$, keeping $\epsilon_\theta$ frozen, by solving

$$\Delta\theta^{\star}=\arg\min_{\Delta\theta}\ \mathbb{E}_{i,\epsilon,t}\big[\lVert\epsilon-\epsilon_{\theta+\Delta\theta}(z_t^{(i)},t,c^{(i)})\rVert_2^2\big],$$

where $z_t^{(i)}$ is the noised latent of $\mathcal{E}(x^{(i)})$. Since $N$ is small, the problem is few-shot, and both the expressiveness of $\Delta\theta$ (rank $r$) and the training amount (epochs) are rate-limiting through the generalization–overfitting trade-off.

## 5. Method

### 5.1 Computing Environment

| Item | Detail |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (12 GB VRAM, Ampere) |
| CPU / RAM | 20 logical cores / 30 GB |
| OS | Ubuntu 26.04 LTS |
| Framework | kohya-ss/sd-scripts, PyTorch 2.5.1 + CUDA 12.4 |
| Python | Python 3.11 via `uv` (system Python 3.14 incompatible) |

### 5.2 Training Data and Selection Criteria

The data was drawn from the author-created sticker set "Masafee" (16 stickers; Figure 1). The character has temporally consistent identifying features: goggles, a crown-patterned bandana, a ringed tail. One severely low-resolution image and two style-inconsistent rough sketches were excluded, yielding the training set $\mathcal{X}$ of **13 images** (all 1000×1000 px RGB).

![Figure 1](../学習データ_13枚.png)

### 5.3 Caption Design and Trigger-Word Binding

The trigger word $\tau=$`masafee` was fixed at the start of every caption, with variable attributes (pose, expression, background, presence of text) enumerated explicitly as tags so they become controllable. Invariant features (goggles, etc.) were left undescribed, residually binding to $\tau$. This concept-binding design follows the DreamBooth [4] lineage. Baked-in text was handled by noting "with text" rather than cropping.

### 5.4 Training Configuration

The optimizer was 8-bit AdamW (quantized optimizer state to reduce VRAM); mixed precision was bfloat16; attention used SDPA. Gradient checkpointing and aspect-ratio bucketing were used, with horizontal-flip augmentation.

## 6. Experiments

**Experiment 1 (Feasibility):** Standard SD1.5, 512 px, $r=32$/$\alpha=16$, learning rate 1e-4, 16 epochs (1040 steps).

**Experiment 2 (Factor sweep):** Six conditions — base model {SD1.5, Counterfeit-V3.0} × learning rate {1e-4, 2e-4} × LoRA rank {16, 32} — trained at 768 px, 30 epochs each, checkpoints at epochs 10/20/30, evaluated with five held-out prompts.

**Experiment 3 (Epoch ablation):** The best condition extended to 60 epochs, comparing epochs 30/40/50/60 (fixed seed).

## 7. Results

### 7.1 Experiment 1: Feasibility

About 9 minutes on the RTX 3060, ~3 GB VRAM. Identifying features were reproduced well and held-out poses were generated (**Q1 confirmed**). Two weaknesses appeared: (i) backgrounds darkened despite "white background", (ii) excessive shading.

### 7.2 Experiment 2: Dominance of the Base Model

The comparison (Figure 2) was clear. SD1.5-base conditions darkened backgrounds and over-shaded; Counterfeit-base conditions gave clean white backgrounds and flat coloring close to the original style. The weaknesses were resolved **by switching the base model, not by tuning learning rate or LoRA rank**. **For Q2, the base model choice is dominant.** This is consistent with the structure of LoRA: learning only a *difference*, the stylistic prior is governed by the base model. The best condition was cf_lr1_d32 (Counterfeit + lr 1e-4 + rank 32).

![Figure 2a](../スイープ比較/sd15_lr1_d32.png)
![Figure 2b](../スイープ比較/cf_lr1_d32.png)

### 7.3 Experiment 3: Performance Plateau over Epochs

Extending to 60 epochs (Figure 3), epochs 30/40/50/60 were nearly indistinguishable; **performance plateaued by epoch 30**. At epochs 50–60, increased saturation signaled mild overfitting. **For Q3, increasing epochs beyond ~30 does not improve quality.**

![Figure 3](../スイープ比較/延長テスト_epoch30-60.png)

### 7.4 Generation with the Final Model

Using the best model (cf_lr1_d32, epoch 30), 12 held-out scenes were generated (Figure 4), preserving identity and the flat style. The rain scene initially darkened; explicitly specifying background luminance resolved it.

![Figure 4](../ショーケース_一覧.png)

## 8. Discussion

**Dominance of the base model.** LoRA learns only $\Delta W$; the stylistic prior is retained in the frozen $W_0$. When target style and prior diverge, a finite-rank $\Delta W$ cannot fully overwrite the style. Choosing a base model consistent with the target style precedes hyperparameter tuning.

**Overfitting and epoch count.** With 13 images, beyond a step threshold $\Delta\theta$ memorizes idiosyncratic configurations; Experiment 3 located this onset near epoch 30.

**Quality ceiling set by data size.** Since epochs plateau, the principal lever for further gains is adding training images. $N=13$ itself sets the effective ceiling.

**Sufficiency of consumer GPUs.** Each run took 9–40 minutes, 3–8 GB VRAM; paid cloud computing is not required for an SD1.5-scale character LoRA.

## 9. Limitations and Future Work

Evaluation was qualitative (visual inspection) without quantitative metrics (CLIP similarity, FID). The training set is a single character of 13 images. Baked-in text was handled by captioning, without a controlled comparison against cropping. Each sweep condition used a single seed. Future work: expanding training images, quantitative metrics, validation with SDXL, and generalization to multiple characters.

## 10. Conclusion

Using only a consumer-grade GPU (RTX 3060), we performed LoRA fine-tuning of the author's original character "Masafee." We showed, with a reproducible procedure, that novel-pose generation with preserved identity is possible from 13 few-shot images (Q1); that output quality depends most strongly on the base model (Q2); and that performance plateaus at about 30 epochs (Q3).

## Appendix A: Key Hyperparameters (cf_lr1_d32)

Base model: Counterfeit-V3.0; resolution: 768×768; LoRA rank/alpha: 32/16; learning rate: 1e-4 (cosine); optimizer: 8-bit AdamW; batch size: 2; epochs: 30; mixed precision: bfloat16; augmentation: horizontal flip; attention: SDPA.

## References

[1] J. Ho, A. Jain, and P. Abbeel. Denoising Diffusion Probabilistic Models. *Advances in Neural Information Processing Systems (NeurIPS)*, 2020.

[2] R. Rombach, A. Blattmann, D. Lorenz, P. Esser, and B. Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2022.

[3] E. J. Hu, Y. Shen, P. Wallis, Z. Allen-Zhu, Y. Li, S. Wang, L. Wang, and W. Chen. LoRA: Low-Rank Adaptation of Large Language Models. *International Conference on Learning Representations (ICLR)*, 2022.

[4] N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, and K. Aberman. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. *IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, 2023.
