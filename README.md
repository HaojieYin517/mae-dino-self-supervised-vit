# MAE → DINOv1 Self-Supervised ViT Workflow (96×96)

This repository contains my implementations and experiment scripts for a two-stage self-supervised learning workflow:
**MAE pretraining → DINOv1 self-distillation**, using a Vision Transformer (ViT) encoder.
Experiments were conducted under moderate-scale constraints (fixed **96×96** resolution and **<100M** parameters)
and evaluated with a **frozen-encoder** protocol.

## What’s included
- **MAE stage**: masked autoencoding pretraining (encoder + lightweight decoder; decoder discarded after pretraining)
- **DINOv1 stage**: self-distillation fine-tuning initialized from the MAE-pretrained encoder
- **Frozen-encoder evaluation**:
  - k-NN (sweeps over K / distance / weighting)
  - linear probing (multinomial logistic regression; selected by validation performance)

## Results (high level)
In this setting, the **MAE → DINOv1** workflow provides a strong initialization for downstream representation quality.
For context, results are discussed in the accompanying report, including comparisons to external baselines
(e.g., DINOv2) where applicable.

## How to run
This repo does not provide a single “one-click” pipeline script. Instead, MAE pretraining, DINO fine-tuning, and
evaluation are run as separate steps via command-line scripts with configurable paths and checkpoints.
See the report and in-folder notes for example commands and expected inputs/outputs.

## Repo structure
- `mae/` — MAE pretraining code (adapted for 96×96, ViT-P8 backbones)
- `dino/` — DINOv1 training code (adapted to load MAE-trained encoders)
- `eval/` — frozen-encoder evaluation utilities (k-NN + linear probe)
- `results/` — project report, checkpoint, and training logs

## Reference
Project report: `results/CVPR_2026_Deep_Learning_Report_final.pdf`

## Acknowledgements
This work builds on established self-supervised learning methods:
- MAE (He et al.)
- DINO (Caron et al.)

(See the report for full citations and experimental details.)