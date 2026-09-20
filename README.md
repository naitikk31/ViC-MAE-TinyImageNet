# ViC-MAE: Finetuning on Tiny-ImageNet-200

Reproduction of the finetuning pipeline from **ViC-MAE** (ECCV 2024) using a pretrained ViT-Base/16 on the Tiny-ImageNet-200 dataset.

**Paper:** [ViC-MAE: Self-Supervised Representation Learning from Images and Video with Contrastive Masked Autoencoders](https://arxiv.org/abs/2303.12001)  
**Original Repo:** [jeffhernandez1995/ViC-MAE](https://github.com/jeffhernandez1995/ViC-MAE)

## Results

| Metric | Value |
|---|---|
| **Top-1 Accuracy** | **80.77%** |
| **Top-5 Accuracy** | **94.39%** |
| Epochs Trained | 33 |
| Dataset | Tiny-ImageNet-200 (200 classes, 100K images) |
| Model | ViT-Base/16 (85.95M params) |
| GPU | NVIDIA Tesla T4 (16 GB) via Kaggle |

### Training Curves
![Training Curves](output/20260918-044542-vit_base_patch16-224/training_curves.png)

## Project Structure

```
├── ViC-MAE/                           # Cloned ViC-MAE repo (with local patches)
├── output/
│   └── 20260918-...-vit_base_patch16-224/
│       ├── log.txt                    # Raw training log (JSON lines)
│       ├── training_curves.png        # Loss & accuracy plots
│       └── training_log.csv           # Per-epoch metrics
├── vicmae_finetune_tinyimagenet.ipynb  # Kaggle notebook (self-contained)
├── plot_results.py                     # Log parser & plotting script
├── setup_tiny_imagenet.py              # Dataset preparation script
├── smoke_test_checkpoint.py            # Checkpoint verification script
├── PROJECT_REPORT.md                   # Full project report
└── README.md                           # This file
```

## How to Reproduce

### Option 1: Kaggle (Recommended)
1. Upload `vicmae_finetune_tinyimagenet.ipynb` to [Kaggle](https://www.kaggle.com)
2. Upload the [ViC-MAE ViT-B/16 checkpoint](https://huggingface.co/jeffhernandez1995/vicmae) as a Kaggle Dataset
3. Enable GPU T4 and Internet
4. Run All cells

### Option 2: Local
```bash
# Create environment
conda create -n vicmae python=3.9 -y
conda activate vicmae
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
pip install timm==0.4.12 wandb matplotlib

# Prepare dataset
python setup_tiny_imagenet.py

# Run finetuning (adjust batch_size for your GPU)
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
    --mixup 0.8 \
    --cutmix 1.0
```

## Key Technical Fixes
- **timm version:** Downgraded to `timm==0.4.12` (newer versions break the forward pass)
- **Eval bug:** Patched `image_finetune.py` line 311 — `epoch` variable was unbound in `--eval` mode
- **Input size:** Set `--input_size 224` to match ViT patch expectations (Tiny-ImageNet is natively 64×64)
- **VRAM optimization:** Used gradient accumulation (batch 16 × accum 8) for 4 GB GPUs

## References
1. Hernandez et al. *ViC-MAE.* ECCV 2024.
2. He et al. *Masked Autoencoders Are Scalable Vision Learners.* CVPR 2022.
3. Dosovitskiy et al. *An Image is Worth 16x16 Words.* ICLR 2021.
