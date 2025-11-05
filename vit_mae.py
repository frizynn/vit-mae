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

class MHSA(nn.Module): # multi-head self-attention
    def __init__(self, embedd_dim, heads=8):
        super().__init__()
        self.heads = heads
        self.dim_per_head = embedd_dim // heads
        self.qkv_projection = nn.Linear(embedd_dim, embedd_dim*3)
        self.output_projection = nn.Linear(embedd_dim, embedd_dim)

    def forward(self, x):
        batch_size, num_patches, embedd_dim = x.shape
        q, k, v = self.qkv_projection(x).chunk(3, dim=-1)
        def reshape(t): # use to split q,k,v into heads
            return t.view(batch_size, num_patches, self.heads, self.dim_per_head).transpose(1, 2) # [batch_size, heads, num_patches, dim_per_head]
        q, k, v = map(reshape, (q, k, v))
        attn = (q @ k.transpose(-2, -1)) / (self.dim_per_head ** 0.5) # compute attention scores
        attn = attn.softmax(dim=-1) # softmax over the last dimension (rows) to get attention weights. each row is a token's attention weights to all other tokens.
        out = (attn @ v).transpose(1, 2).reshape(batch_size, num_patches, embedd_dim)
        return self.output_projection(out)

class TransformerBlock(nn.Module):
    def __init__(self, d, heads=8, mlp_mult=4):
        super().__init__()
        self.pre_attention_norm = nn.LayerNorm(d)
        self.self_attention = MHSA(d, heads)
        self.pre_mlp_norm = nn.LayerNorm(d)
        self.feed_forward = MLP(d, mlp_mult)
    def forward(self, x):
        x = x + self.self_attention(self.pre_attention_norm(x)) # obtain context enriched embeddings
        x = x + self.feed_forward(self.pre_mlp_norm(x))
        return x


class Encoder(nn.Module):
    def __init__(self, patch_dim=768, d_e=1024, depth=12, heads=16, npos=196):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, d_e) # project the patches to the embedding dimension
        self.pos_e = nn.Parameter(torch.zeros(1, npos, d_e))
        self.blocks = nn.ModuleList([TransformerBlock(d_e, heads) for _ in range(depth)]) # transformer blocks concatenated 

    def forward(self, patches, visible_indices):
        # embed patches and add positional encodings for visible patches only
        tokens = self.patch_embed(patches) + self.pos_e[:, visible_indices, :]
        visible_tokens = tokens
        for block in self.blocks: # each block receives the output of the previous block
            visible_tokens = block(visible_tokens)
        return visible_tokens  

class Decoder(nn.Module):
    def __init__(self, d_embedd=1024, d_decoder=512, depth=8, heads=8, npos=196, patch_dim=768): # d_embedd is the embedding dimension of the encoder, d_decoder is the embedding dimension of the decoder, depth is the number of transformer blocks, heads is the number of attention heads for each transformer block, npos is the number of patches in the image, patch_dim is the dimension of the original flattened patches (e.g., 3*16*16=768 for RGB 16x16 patches)
        super().__init__()
        self.proj = nn.Linear(d_embedd, d_decoder) # project the encoder output to the decoder embedding dimension
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_decoder))
        self.pos_d = nn.Parameter(torch.zeros(1, npos, d_decoder))
        self.blocks = nn.ModuleList([TransformerBlock(d_decoder, heads) for _ in range(depth)])
        self.head = nn.Linear(d_decoder, patch_dim) # project the decoder output to the original patch dimension

    def forward(self, visible_embeddings, visible_indices, masked_indices, restore_indices): # visible_embeddings: embeddings of non-masked patches from encoder, visible_indices: indices of the non-masked patches, masked_indices: indices of the masked patches, restore_indices: indices to restore original spatial order
        batch_size = visible_embeddings.size(0) 
        projected_embeddings = self.proj(visible_embeddings)  # project from encoder dim (1024) to decoder dim (512)                  
        num_patches = len(visible_indices) + len(masked_indices)
        full_sequence = projected_embeddings.new_zeros(batch_size, num_patches, projected_embeddings.size(-1))
        full_sequence[:, visible_indices, :] = projected_embeddings
        full_sequence[:, masked_indices, :] = self.mask_token.expand(batch_size, len(masked_indices), -1)
        full_sequence = full_sequence[:, restore_indices, :] + self.pos_d  # restore original spatial order and add positional embeddings
        decoder_input = full_sequence
        for block in self.blocks:
            decoder_input = block(decoder_input)
        return self.head(decoder_input)                         

def mae_loss(pred, target_patches, mask_idx, norm_pix_loss=True):
    """
    Compute MAE loss on masked patches.
    
    Args:
        pred: predicted patches [batch, num_patches, patch_dim]
        target_patches: target patches [batch, num_patches, patch_dim]
        mask_idx: indices of masked patches
        norm_pix_loss: if True, normalize each patch by its mean and variance (as in MAE paper)
    
    Returns:
        loss value
    """
    if norm_pix_loss:
        
        target_mean = target_patches[:, mask_idx, :].mean(dim=-1, keepdim=True)
        target_var = target_patches[:, mask_idx, :].var(dim=-1, keepdim=True, unbiased=False)
        target_normalized = (target_patches[:, mask_idx, :] - target_mean) / (target_var + 1e-6).sqrt()
        
        pred_mean = pred[:, mask_idx, :].mean(dim=-1, keepdim=True)
        pred_var = pred[:, mask_idx, :].var(dim=-1, keepdim=True, unbiased=False)
        pred_normalized = (pred[:, mask_idx, :] - pred_mean) / (pred_var + 1e-6).sqrt()
        
        return ((pred_normalized - target_normalized) ** 2).mean()
    else:
        return ((pred[:, mask_idx, :] - target_patches[:, mask_idx, :])**2).mean()

class ViTMAE(nn.Module):
    """Complete ViT-MAE model combining Encoder and Decoder"""
    def __init__(self, img_size=224, patch_size=16, channels=3, 
                 patch_dim=768, d_e=1024, d_decoder=512, 
                 encoder_depth=12, decoder_depth=8, 
                 encoder_heads=16, decoder_heads=8, 
                 keep_ratio=0.25, norm_pix_loss=True):
        super().__init__()
        self.patch_size = patch_size
        self.channels = channels
        self.keep_ratio = keep_ratio
        self.norm_pix_loss = norm_pix_loss
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = patch_size * patch_size * channels
        
        self.encoder = Encoder(patch_dim=self.patch_dim, d_e=d_e, 
                               depth=encoder_depth, heads=encoder_heads, 
                               npos=self.num_patches)
        self.decoder = Decoder(d_embedd=d_e, d_decoder=d_decoder, 
                               depth=decoder_depth, heads=decoder_heads, 
                               npos=self.num_patches, patch_dim=self.patch_dim)
    
    def forward(self, x):
        # x: [batch, channels, height, width]
        # 1. patchify
        patches = patchify(x, self.patch_size)  # [batch, num_patches, patch_dim]
        
        # 2. random masking
        keep, mask, restore = random_masking(self.num_patches, self.keep_ratio)
        keep = keep.to(x.device)
        mask = mask.to(x.device)
        restore = restore.to(x.device)
        
        # 3. encode visible patches
        visible_patches = patches[:, keep, :]
        visible_embeddings = self.encoder(visible_patches, keep)
        
        # 4. decode all patches
        reconstructed_patches = self.decoder(visible_embeddings, keep, mask, restore)
        
        return reconstructed_patches, patches, mask
