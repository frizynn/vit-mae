import torch, torch.nn as nn
from torch import Tensor

# Patchify 
def patchify(x: Tensor, p: int = 16) -> Tensor:
    batch, channels, height, width = x.shape # e.g. 
    assert height % p == 0 and width % p == 0
    unfold = nn.Unfold(kernel_size=p, stride=p) # e.g [1, 3, 224, 224] --> [1, 3*p*p, (224/p)*(224/p)]
    patches = unfold(x).transpose(1, 2)  # each row is a patch --> [1, (224/p)*(224/p), 3*p*p], because 3*p*p is the flattened patch and we want it as a row
    return patches

# Unpatchify
def unpatchify(patches: Tensor, p: int = 16, height: int = 224, width: int = 224) -> Tensor:
    batch, num_patches, patch_size = patches.shape # e.g. [1, (224/p)*(224/p), 3*p*p]
    fold = nn.Fold(output_size=(height, width), kernel_size=p, stride=p) # e.g. [1, 3*p*p, (224/p)*(224/p)] --> [1, 3, 224, 224]
    x = fold(patches.transpose(1, 2))
    return x

# Random masking
def random_masking(num_patches: int, keep_ratio: float = 0.25):
    perm = torch.randperm(num_patches) # e.g. [0, 1, 2, 3, ..., num_patches-1]
    keep = perm[:int(num_patches*keep_ratio)] 
    mask = perm[int(num_patches*keep_ratio):] 
    restore = torch.argsort(perm) 
    return keep, mask, restore

# Transformer blocks
class MLP(nn.Module):
    def __init__(self, d, hidden_mult=4):
        super().__init__()
        self.up_projection = nn.Linear(d, d*hidden_mult)
        self.activation = nn.GELU()
        self.down_projection = nn.Linear(d*hidden_mult, d)
    def forward(self, x):
        return self.down_projection(self.activation(self.up_projection(x)))

class MHSA(nn.Module):
    def __init__(self, d, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_per_head = d // heads
        self.qkv_projection = nn.Linear(d, d*3)
        self.output_projection = nn.Linear(d, d)

    def forward(self, x):
        B, N, D = x.shape
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)
        def reshape(t):  # [B, N, heads, dim_per_head]
            return t.view(B, N, self.heads, self.dim_per_head).transpose(1, 2)
        q, k, v = map(reshape, (q, k, v))
        attn = (q @ k.transpose(-2, -1)) / (self.dim_per_head ** 0.5)
        attn = attn.softmax(dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.output_projection(out)

class TransformerBlock(nn.Module):
    def __init__(self, d, heads=8, mlp_mult=4):
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(d)
        self.self_attention = MHSA(d, heads)
        self.pre_mlp_norm = nn.LayerNorm(d)
        self.feed_forward = MLP(d, mlp_mult)
    def forward(self, x):
        x = x + self.self_attention(self.pre_attention_norm(x))
        x = x + self.feed_forward(self.pre_mlp_norm(x))
        return x


# class Encoder(nn.Module):
#     def __init__(self, patch_dim=768, d_e=1024, depth=12, heads=16, npos=196):
#         super().__init__()
#         self.patch_embed = nn.Linear(patch_dim, d_e)
#         self.pos_e = nn.Parameter(torch.zeros(1, npos, d_e))
#         self.blocks = nn.ModuleList([TransformerBlock(d_e, heads) for _ in range(depth)])

#     def forward(self, patches, keep_idx):
#         tokens = self.patch_embed(patches) + self.pos_e 
#         x = tokens[:, keep_idx, :]                      
#         for blk in self.blocks:
#             x = blk(x)
#         return x  

# class Decoder(nn.Module):
#     def __init__(self, d_e=1024, d_d=512, depth=8, heads=8, npos=196, patch_dim=768):
#         super().__init__()
#         self.proj = nn.Linear(d_e, d_d)
#         self.mask_token = nn.Parameter(torch.zeros(1, 1, d_d))
#         self.pos_d = nn.Parameter(torch.zeros(1, npos, d_d))
#         self.blocks = nn.ModuleList([TransformerBlock(d_d, heads) for _ in range(depth)])
#         self.head = nn.Linear(d_d, patch_dim)

#     def forward(self, z_vis, keep_idx, mask_idx, restore_idx):
#         B = z_vis.size(0)
#         z_vis = self.proj(z_vis)                    
#         N = len(keep_idx) + len(mask_idx)
#         full = z_vis.new_zeros(B, N, z_vis.size(-1))
#         full[:, keep_idx, :] = z_vis
#         full[:, mask_idx, :] = self.mask_token.expand(B, len(mask_idx), -1)
#         full = full[:, restore_idx, :] + self.pos_d 
#         x = full
#         for blk in self.blocks:
#             x = blk(x)
#         return self.head(x)                         

# def mae_loss(pred, target_patches, mask_idx):
#     return ((pred[:, mask_idx, :] - target_patches[:, mask_idx, :])**2).mean()
