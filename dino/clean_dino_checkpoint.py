import torch
import argparse
from collections import OrderedDict
import numpy as np
import torch.serialization

# Allow safe loading (PyTorch 2.6+)
torch.serialization.add_safe_globals([np.core.multiarray.scalar])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True,
                        help="Original DINO checkpoint (e.g. base_checkpoint0170.pth)")
    parser.add_argument("--out", type=str, required=True,
                        help="Output clean encoder checkpoint")
    args = parser.parse_args()

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # 1. Extract student state dict
    if "student" not in ckpt:
        raise RuntimeError("Checkpoint does not contain 'student' key")

    student_state = ckpt["student"]

    # 2. Strip DistributedDataParallel + keep backbone only
    encoder_state = OrderedDict()
    kept = 0

    for k, v in student_state.items():
        if k.startswith("module.backbone."):
            new_k = k[len("module.backbone."):]  # strip prefix
            encoder_state[new_k] = v
            kept += v.numel()

    print(f"✔ Extracted encoder parameters: {kept:,}")

    # 3. Save clean checkpoint
    torch.save(encoder_state, args.out)

    print("\n✅ Saved clean encoder checkpoint:")
    print(f"   → {args.out}")
    print("   (student backbone only, no teacher / head / optimizer)")

if __name__ == "__main__":
    main()

# usage example
# python clean_dino_checkpoint.py \
#     --ckpt checkpoint0150.pth \
#     --out checkpoint0150_encoder.pth
