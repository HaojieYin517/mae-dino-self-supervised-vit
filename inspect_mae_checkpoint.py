import argparse
import torch

torch.serialization.add_safe_globals([argparse.Namespace])

ENCODER_PREFIXES = [
    "patch_embed",
    "blocks",
    "cls_token",
    "pos_embed",
    "norm",
]

DECODER_PREFIXES = [
    "mask_token",
    "decoder",
]

def is_encoder_key(k):
    return any(k.startswith(p) for p in ENCODER_PREFIXES)

def is_decoder_key(k):
    return any(k.startswith(p) for p in DECODER_PREFIXES)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    args = parser.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ckpt["model"]

    enc_params = 0
    dec_params = 0
    for k, v in state.items():
        if is_encoder_key(k):
            enc_params += v.numel()
        elif is_decoder_key(k):
            dec_params += v.numel()

    total = sum(v.numel() for v in state.values())

    print(f"Encoder parameters: {enc_params:,}")
    print(f"Decoder parameters: {dec_params:,}")
    print(f"Total parameters:   {total:,}")

if __name__ == "__main__":
    main()
    # python inspect_mae_checkpoint.py --ckpt /workspace/mae_base_patch8/checkpoint-20.pth