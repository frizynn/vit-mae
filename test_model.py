import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import yaml
from vit_mae import ViTMAE, unpatchify, mae_loss

def denormalize_image(img_tensor):
    """Denormalize ImageNet-normalized tensor back to [0, 1]"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return img_tensor * std + mean

def load_image(image_path, img_size=224):
    """Load and preprocess image"""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
    
    # Normalize with ImageNet stats
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    
    return img_tensor, img

def visualize_masked_image(original_img, mask_indices, patch_size, img_size):
    """Create visualization showing which patches are masked"""
    # Start with original image
    masked_img = original_img.clone()
    
    # Calculate patches per side
    patches_per_side = img_size // patch_size
    
    # Create mask visualization by overlaying red on masked patches
    for mask_idx in mask_indices:
        # Calculate patch position
        row = mask_idx // patches_per_side
        col = mask_idx % patches_per_side
        
        # Calculate pixel coordinates
        y_start = row * patch_size
        y_end = y_start + patch_size
        x_start = col * patch_size
        x_end = x_start + patch_size
        
        # Overlay red tint on masked patches (blend with original)
        # Keep 50% of original, add 50% red
        masked_img[:, 0, y_start:y_end, x_start:x_end] = (
            masked_img[:, 0, y_start:y_end, x_start:x_end] * 0.5 + 0.5  # Red channel
        )
        masked_img[:, 1, y_start:y_end, x_start:x_end] *= 0.5  # Green channel
        masked_img[:, 2, y_start:y_end, x_start:x_end] *= 0.5  # Blue channel
    
    return masked_img.squeeze(0)  # Remove batch dimension

def test_model(checkpoint_path, image_path, config_path='config.yaml', device=None):
    """Test model on a single image"""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    training_config = config.get('training', {})
    img_size = training_config.get('img_size', 224)
    mask_ratio = float(training_config.get('mask_ratio', 0.75))
    keep_ratio = 1.0 - mask_ratio
    
    # load model architecture configuration
    model_config = training_config.get('model', {})
    patch_size = model_config.get('patch_size', 16)
    channels = model_config.get('channels', 3)
    d_e = model_config.get('d_e', 1024)
    d_decoder = model_config.get('d_decoder', 512)
    encoder_depth = model_config.get('encoder_depth', 12)
    decoder_depth = model_config.get('decoder_depth', 8)
    encoder_heads = model_config.get('encoder_heads', 16)
    decoder_heads = model_config.get('decoder_heads', 8)
    norm_pix_loss = model_config.get('norm_pix_loss', True)
    
    # load model
    model = ViTMAE(
        img_size=img_size,
        patch_size=patch_size,
        channels=channels,
        d_e=d_e,
        d_decoder=d_decoder,
        encoder_depth=encoder_depth,
        decoder_depth=decoder_depth,
        encoder_heads=encoder_heads,
        decoder_heads=decoder_heads,
        keep_ratio=keep_ratio,
        norm_pix_loss=norm_pix_loss
    ).to(device)
    
    # load checkpoint
    if checkpoint_path.endswith('_best.pth') or checkpoint_path.endswith('_final.pth'):
        # load only state dict
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Loaded model from: {checkpoint_path}")
    else:
        # load full checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from: {checkpoint_path}")
        print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
        print(f"  Step: {checkpoint.get('step', 'N/A')}")
        print(f"  Train Loss: {checkpoint.get('train_loss', 'N/A'):.4f}")
        print(f"  Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    model.eval()
    
    # load image
    img_tensor, original_img = load_image(image_path, img_size)
    img_tensor = img_tensor.unsqueeze(0).to(device)  # add batch dimension
    
    # forward pass
    with torch.no_grad():
        reconstructed_patches, original_patches, mask_indices = model(img_tensor)
        
        # calculate loss
        loss = mae_loss(reconstructed_patches, original_patches, mask_indices, norm_pix_loss=model.norm_pix_loss)
      
        num_patches = original_patches.shape[1]
        all_indices = torch.arange(num_patches, device=mask_indices.device)
        # create a boolean mask: true for visible patches, false for masked
        mask_bool = torch.zeros(num_patches, dtype=torch.bool, device=mask_indices.device)
        mask_bool[mask_indices] = True
        keep_indices = all_indices[~mask_bool]
        
        # for visualization, use original patches for visible, reconstructed for masked
        visualization_patches = original_patches.clone()
        visualization_patches[:, mask_indices, :] = reconstructed_patches[:, mask_indices, :]
        
        # unpatchify to get images
        reconstructed_img = unpatchify(visualization_patches, patch_size, img_size, img_size)
        
        # create masked image visualization
        masked_img = visualize_masked_image(img_tensor, mask_indices, patch_size, img_size)
        
        # denormalize
        original_denorm = denormalize_image(img_tensor.squeeze(0))
        reconstructed_denorm = denormalize_image(reconstructed_img.squeeze(0))
        masked_denorm = denormalize_image(masked_img.squeeze(0))
    
    # convert to numpy for plotting
    original_np = original_denorm.cpu().permute(1, 2, 0).numpy()
    reconstructed_np = reconstructed_denorm.cpu().permute(1, 2, 0).numpy()
    masked_np = masked_denorm.cpu().permute(1, 2, 0).numpy()
    
    # clip to [0, 1]
    original_np = np.clip(original_np, 0, 1)
    reconstructed_np = np.clip(reconstructed_np, 0, 1)
    masked_np = np.clip(masked_np, 0, 1)
    
    # calculate metrics
    mse = nn.MSELoss()(reconstructed_denorm, original_denorm).item()
    num_masked = len(mask_indices)
    num_patches = original_patches.shape[1]
    mask_percentage = (num_masked / num_patches) * 100
    
    # plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(masked_np)
    axes[1].set_title(f'Masked Image ({mask_percentage:.1f}% masked)')
    axes[1].axis('off')
    
    axes[2].imshow(reconstructed_np)
    axes[2].set_title(f'Reconstruction\nLoss: {loss.item():.4f} | MSE: {mse:.4f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    
    # save figure
    output_path = Path('test_output.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nResults saved to: {output_path}")
    print(f"MAE Loss (masked patches only): {loss.item():.4f}")
    print(f"MSE (all pixels): {mse:.4f}")
    print(f"Masked patches: {num_masked}/{num_patches} ({mask_percentage:.1f}%)")
    
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test MAE model on an image')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to test image')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Path to config file')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    device = args.device
    if device:
        device = torch.device(device)
    
    test_model(args.checkpoint, args.image, args.config, device)

