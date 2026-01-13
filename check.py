import torch
import os
from vision_transformer import vit_small     # same model used in your run

CHECKPOINT = "/workspace/dino_output_small_299/checkpoint.pth"   # <--- modify if needed

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    assert os.path.isfile(CHECKPOINT), f"❌ Checkpoint not found: {CHECKPOINT}"

    print(f"Loading checkpoint: {CHECKPOINT}")
    ckpt = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)

    # DINO checkpoint stores student + teacher
    student_state = ckpt["student"] if "student" in ckpt else ckpt

    model = vit_small(patch_size=16)
    missing, unexpected = model.load_state_dict(student_state, strict=False)

    print("\n==== STATE LOAD CHECK ====")
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)

    total_params = count_params(model)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("\n===============================")
    print(f"💡 TOTAL parameters      = {total_params:,}")
    print(f"🟢 Trainable parameters  = {trainable_params:,}")
    print("===============================")

    # size on disk
    file_size = os.path.getsize(CHECKPOINT) / (1024**2)
    print(f"📦 Checkpoint filesize   = {file_size:.2f} MB\n")

if __name__ == "__main__":
    main()