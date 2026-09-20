"""
Smoke test: load ViC-MAE ViT-B/16 checkpoint and run one forward pass.
Run from: N:\Academics\Sem 5\AI\Project\
Usage: python smoke_test_checkpoint.py
"""
import sys
import torch

CHECKPOINT = "N:/Academics/Sem 5/AI/Project/vicmae_pretrain_vit_base_in1k_k400.pth"
REPO_DIR   = "N:/Academics/Sem 5/AI/Project/ViC-MAE"
NUM_CLASSES = 200   # Tiny-ImageNet

sys.path.insert(0, REPO_DIR)
import models_vit

print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Build model
print("\nBuilding vit_base_patch16 ...")
model = models_vit.vit_base_patch16(
    num_classes=NUM_CLASSES,
    drop_path_rate=0.1,
    global_pool=False,
)

# Load checkpoint
print(f"Loading checkpoint: {CHECKPOINT}")
ckpt = torch.load(CHECKPOINT, map_location="cpu")

# Checkpoint may be a dict with 'model' key (MAE-style)
if "model" in ckpt:
    state_dict = ckpt["model"]
    print(f"  Found 'model' key with {len(state_dict)} tensors")
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
    print(f"  Found 'state_dict' key with {len(state_dict)} tensors")
else:
    state_dict = ckpt
    print(f"  Flat checkpoint with {len(state_dict)} tensors")

print(f"  First 5 keys: {list(state_dict.keys())[:5]}")

# Use the repo's interpolate_pos_embed utility
sys.path.insert(0, REPO_DIR)
from util.pos_embed import interpolate_pos_embed
interpolate_pos_embed(model, state_dict)

# Load — allow missing/unexpected (head will be missing for pretrain ckpt)
msg = model.load_state_dict(state_dict, strict=False)
print(f"\nLoad result:")
print(f"  Missing keys:    {msg.missing_keys}")
print(f"  Unexpected keys: {msg.unexpected_keys[:5]} ...")

# Forward pass on dummy batch
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()
x = torch.randn(2, 3, 224, 224, device=device)
with torch.no_grad():
    out = model(x)

print(f"\nForward pass OK!")
print(f"  Input shape:  {x.shape}")
print(f"  Output shape: {out.shape}  (expected [2, {NUM_CLASSES}])")
assert out.shape == (2, NUM_CLASSES), f"Shape mismatch: {out.shape}"
print("\n[SMOKE TEST PASSED]")
