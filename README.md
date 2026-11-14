# ViC-MAE: Finetuning on Tiny-ImageNet-200

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?style=flat&logo=pytorch)](https://pytorch.org)
[![Model](https://img.shields.io/badge/Model-ViT--Base%2F16-blue.svg)](https://arxiv.org/abs/2010.11929)
[![Paper](https://img.shields.io/badge/Paper-ECCV%202024-green.svg)](https://arxiv.org/abs/2303.12001)
[![Top-1 Accuracy](https://img.shields.io/badge/Top--1%20Acc-80.77%25-brightgreen.svg)](#results)
[![Top-5 Accuracy](https://img.shields.io/badge/Top--5%20Acc-94.39%25-brightgreen.svg)](#results)

A complete reproduction and adaptation of the finetuning pipeline from **ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders** (ECCV 2024).

This repository adapts a pretrained Vision Transformer (`ViT-Base/16`, 85.95M parameters) originally pretrained on ImageNet-1K + Kinetics-400, and finetunes it for 200-class fine-grained image classification on the **Tiny-ImageNet-200** benchmark.

---

## 📌 Key Highlights & Results

- **80.77% Top-1 Accuracy** and **94.39% Top-5 Accuracy** on Tiny-ImageNet validation split.
- **Fast transfer learning convergence:** Climbed from 12.39% to >60% accuracy within the first 3 epochs.
- **Interactive Web Demo (`demo.py`):** Offline image uploads, top-five predictions, and a searchable Tiny-ImageNet class list.
- **Dual execution paths:** Preconfigured for both high-memory cloud training (Kaggle dual-T4) and constrained local GPUs (RTX 3050 4GB with gradient accumulation).

### Performance Summary

| Metric | Result |
| :--- | :--- |
| **Model Architecture** | Vision Transformer (`vit_base_patch16`, 16×16 patch size) |
| **Parameters** | 85.95 Million |
| **Target Dataset** | Tiny-ImageNet-200 (100,000 train / 10,000 validation images) |
| **Input Resolution** | 224 × 224 (bicubic interpolation & crop) |
| **Finetuned Epochs** | 33 epochs (Cosine LR schedule with 5-epoch warmup) |
| **Effective Batch Size** | 128 |
| **Best Top-1 Accuracy** | **80.77%** |
| **Best Top-5 Accuracy** | **94.39%** |
| **Final Validation Loss** | 0.839 |

---

## 📈 Training & Validation Curves

The model was finetuned with AdamW, layer-wise learning rate decay (0.75), Mixup (0.8), CutMix (1.0), and label smoothing (0.1).

![Training & Validation Curves](output/20260918-044542-vit_base_patch16-224/training_curves.png)

*Validation accuracy rises smoothly while validation loss steadily drops from 4.20 to 0.84 with no divergence or overfitting.*

---

## 🔬 Interactive Web Demo (`demo.py`)

A local Flask app that ranks the 200 Tiny-ImageNet classes for an uploaded image. It selects a local finetuning checkpoint by default, identifies the loaded file, and uses the validation preprocessing. If a matching evaluation record exists, the header shows its measured accuracy.

With the matching `checkpoint-20.pth`, the demo uses six test-time views and temperature-scaled probabilities. The low-confidence warning uses calibrated maximum softmax probability, and the separate outside-classes flag uses raw energy from the mean logits. The **Show heatmap** control compares the center crop, Grad-CAM heatmap, and overlay side by side. Other checkpoints use uncalibrated scores unless matching reliability settings are supplied.

- **Theme transition:** A wavy edge and silk strands travel down when switching to dark mode and up when switching to light mode. Reduced-motion mode switches instantly.
- **Image upload:** Drag and drop, browse, or try a local validation example. The demo limits uploads to 8 MiB and 20 megapixels.
- **Top-five results:** The demo uses CUDA when available and shows ranked classes with their model scores.

### Running the Demo

With the demo's Python 3.11 or newer environment active, run:

```bash
python demo.py
```
Then open [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.

From the project directory on Windows, the local interpreter is `.\.venv-vicmae-cached\env\Scripts\python.exe`.

Use `python demo.py --checkpoint PATH` to select a checkpoint. Run `python eval_tiny_imagenet.py` to evaluate the default checkpoint on the validation split. The evaluator saves a new `validation.json` with the actual image count and metrics under `output/classification_eval/`.

The demo requires the extracted Tiny-ImageNet dataset, the upstream `ViC-MAE/` checkout, and local classifier weights. To explicitly use the settings shipped for the measured checkpoint, run:

```bash
python demo.py --checkpoint output/20260918-044542-vit_base_patch16-224/checkpoint-20.pth --reliability-settings extension/demo_settings.json
```

The app verifies the checkpoint hash before applying those settings.

---

## 🛠️ Technical Fixes & Engineering Overcomes

During reproduction, several issues in the upstream codebase and environment were identified and resolved:

1. **`timm` Library Breaking Changes:**
   * *Issue:* Overridden `VisionTransformer.forward_features()` in `models_vit.py` broke under `timm >= 1.0` due to added keyword argument `attn_mask`.
   * *Fix:* Pinned dependency strictly to `timm==0.4.12` to maintain API compatibility.
2. **Upstream Standalone Evaluation Bug:**
   * *Issue:* Running `image_finetune.py` with `--eval` crashed with `UnboundLocalError` because `epoch` was only defined inside the training loop.
   * *Fix:* Patched line 311 to set `epoch=0` in eval mode.
3. **Resolution & Patch Embedding Mismatch:**
   * *Issue:* Tiny-ImageNet images are 64×64 native, but `ViT-Base/16` patch projection requires 224×224 (14×14 patch grid). Passing `--input_size 64` triggers patch assertion failures.
   * *Fix:* Standardized `--input_size 224` and utilized `RandomResizedCropAndInterpolation` upscaling.
4. **VRAM Optimization for Consumer Hardware:**
   * *Issue:* Running batch size 64 exceeded 4 GB VRAM on mobile/desktop GPUs (RTX 3050), triggering Windows shared memory swapping.
   * *Fix:* Parameterized gradient accumulation (`--batch_size 16 --accum_iter 8`) to maintain the effective batch size of 128 while capping memory at 3.08 GB.

---

## 🚀 How to Reproduce

### Option A: Cloud Training on Kaggle (Recommended)

1. Clone or import `vicmae_finetune_tinyimagenet.ipynb` into a Kaggle Notebook.
2. Attach the [ViC-MAE pretrained checkpoint](https://huggingface.co/jeffhernandez1995/vicmae) as an input dataset.
3. Select **GPU T4 × 2** accelerator and enable Internet access.
4. Run the notebook top-to-bottom. Checkpoints and evaluation logs will be generated in `/kaggle/working/output`.

### Option B: Local Training

```bash
# 1. Clone repository
git clone https://github.com/naitikk31/ViC-MAE-TinyImageNet.git
cd ViC-MAE-TinyImageNet

# 2. Set up Python environment
conda create -n vicmae python=3.9 -y
conda activate vicmae
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install timm==0.4.12 matplotlib flask

# 3. Clone upstream ViC-MAE repository
git clone https://github.com/jeffhernandez1995/ViC-MAE ViC-MAE

# 4. Download and structure Tiny-ImageNet
python setup_tiny_imagenet.py

# 5. Download ViC-MAE checkpoint
# Download vicmae_pretrain_vit_base_in1k_k400.pth from official repository/HF

# 6. Launch finetuning
cd ViC-MAE
python image_finetune.py \
    --model vit_base_patch16 \
    --finetune ../vicmae_pretrain_vit_base_in1k_k400.pth \
    --data_path ../tiny-imagenet-200 \
    --nb_classes 200 \
    --epochs 40 \
    --batch_size 128 \
    --input_size 224 \
    --blr 1e-3 \
    --layer_decay 0.75 \
    --weight_decay 0.05 \
    --drop_path 0.1 \
    --mixup 0.8 \
    --cutmix 1.0 \
    --output_dir ../output
```

---

## 📂 Repository Layout

```
.
├── demo.py                             # Interactive Flask classifier demo
├── vicmae_finetune_tinyimagenet.ipynb  # End-to-end self-contained Kaggle notebook
├── plot_results.py                     # Metric extraction & curve plotting script
├── setup_tiny_imagenet.py              # Automated download & ImageFolder reorganizer
├── smoke_test_checkpoint.py            # Checkpoint loader & tensor shape verification
├── output/
│   └── 20260918-044542-vit_base_patch16-224/
│       ├── log.txt                     # JSON lines containing raw training telemetry
│       ├── training_log.csv            # Tabular training/validation loss & accuracy
│       └── training_curves.png         # Loss and accuracy curve charts
├── .gitignore                          # Excludes checkpoints, raw datasets & private reports
└── README.md
```

---

## 📚 References

1. **ViC-MAE:** Hernandez, J., Villegas, R., & Ordonez, V. (2024). *ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders.* European Conference on Computer Vision (ECCV 2024). [arXiv:2303.12001](https://arxiv.org/abs/2303.12001)
2. **Masked Autoencoders:** He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). *Masked Autoencoders Are Scalable Vision Learners.* IEEE/CVF CVPR 2022.
3. **Vision Transformer (ViT):** Dosovitskiy, A., et al. (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021.
