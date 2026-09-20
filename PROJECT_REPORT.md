# ViC-MAE: Reproducing Vision Transformer Finetuning on Tiny-ImageNet-200

**Course:** Artificial Intelligence — Semester 5  
**Paper:** ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders (ECCV 2024)  
**Authors (Original):** Jefferson Hernandez, Ruben Villegas, Vicente Ordonez  
**Repository:** https://github.com/jeffhernandez1995/ViC-MAE

---

## 1. Introduction

This project reproduces the **finetuning pipeline** of ViC-MAE, a state-of-the-art self-supervised learning method published at ECCV 2024. The goal was to take a Vision Transformer (ViT-Base/16) that was pretrained using the ViC-MAE framework on ImageNet-1K and Kinetics-400, and finetune it for **image classification** on the Tiny-ImageNet-200 dataset.

**Key Contributions of This Project:**
- Successfully adapted the ViC-MAE codebase to work with Tiny-ImageNet-200 (200 classes, 100K training images)
- Identified and fixed a bug in the original repository's evaluation code
- Resolved multiple dependency and hardware compatibility issues
- Achieved **80.77% Top-1 accuracy** and **94.39% Top-5 accuracy** after 33 epochs of finetuning

---

## 2. Background

### 2.1 Masked Autoencoders (MAE)
Masked Autoencoders are a self-supervised pretraining strategy for Vision Transformers. The core idea is simple: randomly mask out 75% of the patches in an input image, and train the model to reconstruct the missing patches. By solving this "visual puzzle" millions of times, the model learns rich, general-purpose representations of visual features like edges, textures, shapes, and spatial relationships — all without requiring any human-labeled data.

### 2.2 ViC-MAE
ViC-MAE extends the MAE framework by incorporating **both images and video** during pretraining, and adding a **contrastive learning objective** alongside the reconstruction loss. This forces the model to learn representations that are not only good at pixel reconstruction but also semantically meaningful across modalities. The result is a pretrained backbone that transfers exceptionally well to downstream tasks.

### 2.3 Vision Transformer (ViT-Base/16)
The backbone architecture used in this project is ViT-Base/16:
- **Patch size:** 16×16 pixels
- **Hidden dimension:** 768
- **Transformer blocks:** 12
- **Attention heads:** 12
- **Total parameters:** 85.95 million
- **Input resolution:** 224×224

The model divides each input image into a grid of 14×14 = 196 patches (each 16×16 pixels), linearly embeds them, adds positional encodings, and processes the sequence through 12 Transformer encoder blocks.

### 2.4 Finetuning (Transfer Learning)
Finetuning takes the pretrained ViT backbone (which has learned general visual features) and adds a new classification head (a linear layer mapping from 768 dimensions to 200 classes). The entire model is then trained end-to-end on the target dataset with a small learning rate, allowing the pretrained features to adapt to the specific classification task.

---

## 3. Dataset: Tiny-ImageNet-200

| Property | Value |
|---|---|
| Number of classes | 200 |
| Training images | 100,000 (500 per class) |
| Validation images | 10,000 (50 per class) |
| Original image size | 64×64 |
| Resized to | 224×224 (for ViT compatibility) |

Tiny-ImageNet-200 is a subset of the full ImageNet (ILSVRC-2012) dataset. It was chosen for this project as a practical compromise — it is small enough to train on limited hardware within reasonable time, while still being challenging enough (200 fine-grained classes) to meaningfully evaluate transfer learning quality.

> **Note:** The original ViC-MAE paper reports results on the full ImageNet-1K dataset (1000 classes, 1.28M images). Direct numerical comparison is therefore not applicable; however, the methodology and pipeline are identical.

**Dataset Preparation:**
The raw Tiny-ImageNet download required restructuring:
1. Training images were nested inside per-class `images/` subdirectories — these were flattened to match PyTorch's `ImageFolder` expectations
2. Validation images were stored in a single flat directory with a separate `val_annotations.txt` mapping file — these were reorganized into per-class subdirectories
3. All images were resized from 64×64 to 224×224 via `RandomResizedCropAndInterpolation` (training) and `Resize + CenterCrop` (validation)

---

## 4. Methodology

### 4.1 Training Configuration

| Hyperparameter | Value |
|---|---|
| Pretrained checkpoint | `vicmae_pretrain_vit_base_in1k_k400.pth` |
| Optimizer | AdamW |
| Base learning rate | 1×10⁻³ |
| Actual learning rate | 5×10⁻⁴ (scaled by effective batch size / 256) |
| Weight decay | 0.05 |
| Layer-wise LR decay | 0.75 |
| Warmup epochs | 5 |
| LR schedule | Cosine annealing (min LR = 1×10⁻⁶) |
| Batch size | 128 |
| Gradient accumulation | 1 |
| Effective batch size | 128 |
| Epochs trained | 33 (out of 40 planned) |
| Mixed precision | FP16 (via `torch.cuda.amp`) |
| Drop path rate | 0.1 |

### 4.2 Data Augmentation

| Augmentation | Setting |
|---|---|
| Random Resized Crop | 224×224, scale (0.08, 1.0) |
| Random Horizontal Flip | p=0.5 |
| RandAugment | `rand-m9-mstd0.5-inc1` |
| Mixup | α=0.8 |
| CutMix | α=1.0 |
| Random Erasing | p=0.25, mode=pixel |
| Label Smoothing | 0.1 |

### 4.3 Hardware

| Phase | Hardware | VRAM | Batch Size | Time per Epoch |
|---|---|---|---|---|
| Smoke test (5 epochs) | NVIDIA RTX 3050 (local) | 4 GB | 16 (accum=8) | ~32 min |
| Full training (33 epochs) | NVIDIA Tesla T4 ×2 (Kaggle) | 16 GB | 128 (accum=1) | ~17 min |

---

## 5. Technical Challenges & Solutions

### 5.1 Timm Library Incompatibility
**Problem:** The ViC-MAE repository's `models_vit.py` overrides `VisionTransformer.forward_features()` from the `timm` library. Modern versions of `timm` (≥1.0) changed the method signature to include an `attn_mask` parameter, causing a `TypeError` crash.  
**Solution:** Downgraded to `timm==0.4.12`, which matches the API the codebase was written against.

### 5.2 Input Size Mismatch
**Problem:** Tiny-ImageNet images are 64×64, but `ViT-Base/16` expects 224×224 inputs. Passing `--input_size 64` caused `PatchEmbed` to crash with an `AssertionError` because the model was initialized with `img_size=224`.  
**Solution:** Set `--input_size 224` and let the data augmentation pipeline handle the upsampling via `RandomResizedCropAndInterpolation`.

### 5.3 VRAM Limitation on Local GPU
**Problem:** The default batch size of 64 exceeded the 4 GB VRAM of the RTX 3050, causing Windows to swap GPU memory to system RAM and reducing training speed from 0.3s/step to 15s/step.  
**Solution:** Reduced batch size to 16 with gradient accumulation of 8, maintaining the same effective batch size of 128 while keeping VRAM usage at 3.08 GB.

### 5.4 Windows DataLoader Hang
**Problem:** PyTorch's multi-process `DataLoader` hung silently on Windows when `num_workers > 0`.  
**Solution:** Set `--num_workers 0` for local training. On Kaggle (Linux), `num_workers=4` worked without issues.

### 5.5 Evaluation Bug in Original Repository
**Problem:** Running standalone evaluation with `--eval` flag caused an `UnboundLocalError` because the variable `epoch` was only defined inside the training loop, but the evaluation code path references it before the loop executes.  
**Solution:** Patched line 311 of `image_finetune.py` to use `epoch=0` in the eval-only code path.

### 5.6 HuggingFace Checkpoint Access
**Problem:** The pretrained checkpoint URL on HuggingFace returned `HTTP 401 Unauthorized` when accessed from Kaggle's servers.  
**Solution:** Downloaded the checkpoint locally and uploaded it as a private Kaggle Dataset, then attached it to the notebook as an input source.

---

## 6. Results

### 6.1 Final Metrics

| Metric | Value |
|---|---|
| **Top-1 Accuracy** | **80.77%** |
| **Top-5 Accuracy** | **94.39%** |
| Final Training Loss | 3.14 |
| Final Validation Loss | 0.84 |
| Epochs Completed | 33 / 40 |
| Total Training Time | ~9.5 hours |

### 6.2 Training Curves

![ViC-MAE Training Curves — Training loss decreasing from 5.13 to 3.14, Top-1 accuracy rising from 12.4% to 80.8%, Top-5 accuracy rising from 36.7% to 94.4%](C:/Users/Naitik's Laptop/.gemini/antigravity/brain/8e544339-c409-4693-8a45-2bbda616cb0d/training_curves.png)

### 6.3 Key Observations

1. **Rapid early learning:** The model jumped from 12.4% to 53.6% Top-1 accuracy in just the first 3 epochs, demonstrating the power of the pretrained ViC-MAE representations.

2. **Warmup phase effect:** During the 5-epoch learning rate warmup (epochs 0–4), accuracy climbed steeply from 12.4% to 62.4%. The cosine annealing schedule then provided smooth, continued improvement.

3. **Diminishing returns after epoch 25:** The accuracy curve flattens significantly after epoch 25 (~79%), with only marginal gains in the final 8 epochs (79% → 80.8%). This confirms that stopping at epoch 33 did not significantly impact final performance.

4. **No overfitting observed:** The validation loss continued to decrease throughout training (from 4.20 to 0.84), with no divergence between training and validation metrics, indicating healthy generalization.

5. **Strong Top-5 performance:** The 94.4% Top-5 accuracy indicates that even when the model's top prediction is wrong, the correct class is almost always in its top 5 guesses — a sign of well-structured learned representations.

---

## 7. Comparison with Related Work

| Method | Dataset | Top-1 Acc |
|---|---|---|
| ViC-MAE ViT-B/16 (paper) | ImageNet-1K (1000 classes) | 83.7% |
| **ViC-MAE ViT-B/16 (this project)** | **Tiny-ImageNet-200 (200 classes)** | **80.8%** |
| MAE ViT-B/16 (He et al., 2022) | ImageNet-1K (1000 classes) | 83.6% |
| Supervised ViT-B/16 (Dosovitskiy et al.) | ImageNet-1K (1000 classes) | 77.9% |

> **Note:** The datasets differ (ImageNet-1K vs Tiny-ImageNet-200), so direct numerical comparison is not meaningful. The comparison is included to show that ViC-MAE's pretrained representations transfer effectively even to a much smaller, lower-resolution dataset.

---

## 8. Conclusion

This project successfully reproduced the finetuning pipeline of ViC-MAE, a state-of-the-art self-supervised vision model. Starting from a pretrained ViT-Base/16 checkpoint, we adapted the codebase to work with the Tiny-ImageNet-200 dataset and achieved **80.77% Top-1 accuracy** — demonstrating that representations learned through masked autoencoding and contrastive learning on large-scale image and video data transfer remarkably well to smaller downstream classification tasks.

The project also identified and resolved several practical engineering challenges, including library incompatibilities, memory optimization for consumer-grade GPUs, and a bug in the original repository's evaluation code.

---

## 9. References

1. Hernandez, J., Villegas, R., & Ordonez, V. (2024). *ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders.* ECCV 2024.
2. He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.
3. Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
4. Vaswani, A., et al. (2017). *Attention Is All You Need.* NeurIPS 2017.

---

## Appendix A: Repository Structure

```
N:\Academics\Sem 5\AI\Project\
├── ViC-MAE/                          # Cloned repository (with local patches)
│   ├── image_finetune.py             # Main finetuning script (patched eval bug)
│   ├── engine_finetune.py            # Training and evaluation loops
│   ├── models_vit.py                 # ViT model definitions
│   └── util/                         # Utilities (logging, LR scheduling, etc.)
├── output/
│   └── 20260918-044542-vit_base_patch16-224/
│       ├── log.txt                   # Raw training log (JSON lines)
│       ├── training_curves.png       # Training/validation curves plot
│       ├── training_log.csv          # Parsed CSV of per-epoch metrics
│       ├── checkpoint-0.pth          # Model checkpoint at epoch 0
│       └── checkpoint-20.pth         # Model checkpoint at epoch 20
├── vicmae_finetune_tinyimagenet.ipynb # Kaggle notebook
├── plot_results.py                   # Log parser and plotting script
├── setup_tiny_imagenet.py            # Dataset preparation script
└── smoke_test_checkpoint.py          # Checkpoint verification script
```

## Appendix B: Per-Epoch Results

| Epoch | Train Loss | Val Top-1 (%) | Val Top-5 (%) |
|-------|-----------|---------------|---------------|
| 0     | 5.132     | 12.39         | 36.73         |
| 1     | 4.711     | 38.39         | 70.55         |
| 2     | 4.363     | 53.63         | 81.44         |
| 3     | 4.192     | 60.61         | 85.44         |
| 4     | 4.104     | 62.36         | 86.20         |
| 5     | 4.018     | 64.94         | 87.36         |
| 6     | 3.948     | 67.18         | 88.35         |
| 7     | 3.878     | 68.22         | 89.26         |
| 8     | 3.816     | 70.09         | 90.21         |
| 9     | 3.764     | 71.42         | 90.41         |
| 10    | 3.738     | 71.22         | 90.60         |
| 11    | 3.687     | 72.73         | 91.18         |
| 12    | 3.672     | 73.06         | 91.19         |
| 13    | 3.612     | 73.62         | 91.51         |
| 14    | 3.603     | 74.14         | 91.76         |
| 15    | 3.510     | 74.70         | 92.01         |
| 16    | 3.540     | 74.83         | 92.21         |
| 17    | 3.477     | 75.77         | 92.48         |
| 18    | 3.480     | 76.47         | 92.97         |
| 19    | 3.440     | 76.59         | 93.06         |
| 20    | 3.401     | 77.24         | 93.31         |
| 21    | 3.381     | 77.84         | 93.43         |
| 22    | 3.340     | 77.85         | 93.34         |
| 23    | 3.350     | 78.36         | 93.58         |
| 24    | 3.312     | 79.06         | 93.90         |
| 25    | 3.286     | 79.14         | 93.73         |
| 26    | 3.271     | 79.51         | 93.71         |
| 27    | 3.222     | 79.68         | 94.06         |
| 28    | 3.255     | 80.14         | 94.24         |
| 29    | 3.184     | 80.39         | 94.14         |
| 30    | 3.217     | 80.25         | 94.22         |
| 31    | 3.192     | 80.74         | 94.50         |
| 32    | 3.143     | 80.77         | 94.39         |
