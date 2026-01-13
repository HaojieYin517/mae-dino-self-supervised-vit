import torch
from models_mae import mae_vit_small_patch8

ckpt_path = "/workspace/mae_patch8_512/checkpoint-0.pth"  # adjust if needed

# Force full pickle loading (safe since it's your own checkpoint)
checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

model = mae_vit_small_patch8()
if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"], strict=False)
else:
    model.load_state_dict(checkpoint, strict=False)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"\nTotal parameters:     {total_params:,}  ({total_params/1e6:.2f}M)")
print(f"Trainable parameters: {trainable_params:,}  ({trainable_params/1e6:.2f}M)")
print("\nUnder 100M:", total_params < 100_000_000)
