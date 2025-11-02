# ViT-MAE: Vision Transformer with Masked Autoencoder

A from-scratch PyTorch implementation of the **Vision Transformer Masked Autoencoder (ViT-MAE)** for self-supervised image representation learning.

## Overview

ViT-MAE combines Vision Transformers with masked autoencoding:
- Divides images into patches and randomly masks 75% of them
- Encoder processes only visible patches
- Decoder reconstructs all patches (visible and masked)
- Learns rich visual representations without labels

## Reference

**"Masked Autoencoders Are Scalable Vision Learners"** (He et al., 2021)  
Paper included as `vit-mae.pdf` in this repository.

## Implementation

Clean PyTorch implementation including:
- Multi-Head Self-Attention (MHSA)
- Transformer blocks with attention and MLP
- Patch masking and reconstruction
- Encoder/Decoder architecture

## Dependencies

- PyTorch >= 1.9.0
- NumPy >= 1.21.0

## Usage

```python
from vit_mae import Encoder, Decoder, mae_loss

encoder = Encoder(patch_dim=768, d_e=1024, depth=12, heads=16)
decoder = Decoder(d_e=1024, d_d=512, depth=8, heads=8)
```
