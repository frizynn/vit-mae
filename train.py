import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from vit_mae import ViTMAE, mae_loss
import yaml
import numpy as np
import random
import logging
import os
from pathlib import Path
from collections import deque
from data_loader import load_datasets_from_config

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

def train_epoch(model, train_loader, optimizer, device, global_step=0, 
                val_loader=None, validate_func=None, scheduler=None,
                validation_frequency='epoch', every_n_steps=None,
                save_every_n_steps=None, checkpoint_dir=None, 
                epoch_num=0, save_checkpoint_func=None, 
                best_val_loss=None, save_best=False):
    """train for one epoch with optional validation and checkpointing by steps"""
    model.train()
    total_loss = 0
    num_batches = 0
    current_step = global_step
    last_step_checkpoint = global_step
    last_step_validation = global_step if validation_frequency == 'step' and every_n_steps else None
    
    for batch_idx, images in enumerate(train_loader):
        images = images.to(device)
        
        # forward pass
        reconstructed_patches, target_patches, mask_indices = model(images)
        
        # compute loss only on masked patches
        loss = mae_loss(reconstructed_patches, target_patches, mask_indices)
        
        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        current_step += 1
        
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Batch {batch_idx + 1}/{len(train_loader)}, Step {current_step}, Loss: {loss.item():.4f}")
        
        # Validate by step if configured
        if validation_frequency == 'step' and every_n_steps is not None and validate_func is not None:
            if current_step - last_step_validation >= every_n_steps:
                logger.info(f"Validating at step {current_step}...")
                val_loss = validate_func(model, val_loader, device)
                logger.info(f"Val Loss at step {current_step}: {val_loss:.4f}")
                last_step_validation = current_step
                
                # Save best model if validation improved
                if save_best and best_val_loss is not None and val_loss < best_val_loss[0]:
                    best_val_loss[0] = val_loss
                    best_model_path = checkpoint_dir / "vit_mae_best.pth"
                    torch.save(model.state_dict(), best_model_path)
                    logger.info(f"Best model saved at step {current_step} (val_loss: {val_loss:.4f})")
        
        # Save checkpoint by step if configured
        if save_every_n_steps is not None and save_checkpoint_func is not None:
            if current_step - last_step_checkpoint >= save_every_n_steps:
                checkpoint_path = checkpoint_dir / f"checkpoint_step_{current_step}.pth"
                # Get current average loss for checkpoint
                current_avg_loss = total_loss / num_batches if num_batches > 0 else 0
                val_loss_for_checkpoint = best_val_loss[0] if best_val_loss is not None else float('inf')
                save_checkpoint_func(model, optimizer, scheduler, epoch_num + 1, current_step, 
                                   current_avg_loss, val_loss_for_checkpoint, checkpoint_path)
                logger.info(f"Step checkpoint saved at step {current_step}: {checkpoint_path}")
                last_step_checkpoint = current_step
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    return avg_loss, current_step

def validate_epoch(model, val_loader, device):
    """validate for one epoch"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for images in val_loader:
            images = images.to(device)
            
            # forward pass
            reconstructed_patches, target_patches, mask_indices = model(images)
            
            # compute loss only on masked patches
            loss = mae_loss(reconstructed_patches, target_patches, mask_indices)
            
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches if num_batches > 0 else 0

def save_checkpoint(model, optimizer, scheduler, epoch, step, train_loss, val_loss, checkpoint_path):
    """Save checkpoint with model state"""
    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)

def cleanup_old_checkpoints(checkpoint_dir, keep_last_n, prefix='checkpoint_'):
    """Remove old checkpoints, keeping only the last N (by modification time)"""
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return
    
    # Get all checkpoint files matching the prefix (excluding best and final models)
    all_checkpoints = list(checkpoint_dir.glob(f"{prefix}*.pth"))
    # Exclude best and final models from cleanup
    checkpoints = [c for c in all_checkpoints if 'best' not in c.name and 'final' not in c.name]
    
    if len(checkpoints) <= keep_last_n:
        return
    
    # Sort by modification time (newest first)
    checkpoints = sorted(checkpoints, key=lambda x: x.stat().st_mtime, reverse=True)
    
    # Keep only the last N checkpoints, remove the rest
    for old_checkpoint in checkpoints[keep_last_n:]:
        old_checkpoint.unlink()
        logger.info(f"Removed old checkpoint: {old_checkpoint}")

def main():
    # load config
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # configuration
    training_config = config.get('training', {})
    img_size = training_config.get('img_size', 224)
    batch_size = training_config.get('batch_size', 8)
    num_workers = training_config.get('num_workers', 2)
    seed = training_config.get('seed', 42)
    num_epochs = training_config.get('num_epochs', 10)
    learning_rate = float(training_config.get('learning_rate', 1e-4))
    mask_ratio = float(training_config.get('mask_ratio', 0.75))
    keep_ratio = 1.0 - mask_ratio  # keep_ratio = visible patches (0.25 for 75% masking)
    
    validation_config = training_config.get('validation', {})
    validation_frequency = validation_config.get('frequency', 'epoch')
    every_n_epochs = validation_config.get('every_n_epochs', 1)
    every_n_steps = validation_config.get('every_n_steps', None)
    
    # checkpoint configuration
    checkpoint_config = training_config.get('checkpoint', {})
    save_best = checkpoint_config.get('save_best', True)
    save_every_n_epochs = checkpoint_config.get('save_every_n_epochs', 1)
    save_every_n_steps = checkpoint_config.get('save_every_n_steps', None)
    keep_last_n = checkpoint_config.get('keep_last_n', 5)
    
    # early stopping configuration
    early_stopping_config = training_config.get('early_stopping', {})
    patience = early_stopping_config.get('patience', None)
    min_delta = early_stopping_config.get('min_delta', 0.0)
    
    # scheduler configuration
    scheduler_config = training_config.get('scheduler', {})
    scheduler_type = scheduler_config.get('type', 'CosineAnnealingWarmRestarts')
    T_0 = int(scheduler_config.get('T_0', 10))
    T_mult = int(scheduler_config.get('T_mult', 2))
    eta_min = float(scheduler_config.get('eta_min', 1e-6))
    
    # set seed for reproducibility
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    patch_size = 16
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger.info(f"Using device: {device}")
    logger.info(f"Image size: {img_size}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Seed: {seed}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Number of epochs: {num_epochs}")
    logger.info(f"Mask ratio: {mask_ratio} (keep ratio: {keep_ratio})")
    logger.info(f"Scheduler: {scheduler_type} (T_0={T_0}, T_mult={T_mult}, eta_min={eta_min})")
    logger.info(f"Validation: frequency={validation_frequency}, every_n_epochs={every_n_epochs}, every_n_steps={every_n_steps}")
    logger.info(f"Checkpoints: save_best={save_best}, save_every_n_epochs={save_every_n_epochs}, save_every_n_steps={save_every_n_steps}, keep_last_n={keep_last_n}")
    
    # Create checkpoint directory
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    
    # load datasets from config
    train_loader, val_loader = load_datasets_from_config(
        config_path='config.yaml',
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers,
        seed=seed
    )
    
    # create model
    model = ViTMAE(
        img_size=img_size,
        patch_size=patch_size,
        channels=3,
        d_e=1024,
        d_decoder=512,
        encoder_depth=12,
        decoder_depth=8,
        encoder_heads=16,
        decoder_heads=8,
        keep_ratio=keep_ratio
    ).to(device)
    
    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.05)
    
    # learning rate scheduler
    if scheduler_type == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    else:
        logger.warning(f"Unknown scheduler type: {scheduler_type}. Using CosineAnnealingWarmRestarts as default.")
        scheduler = CosineAnnealingWarmRestarts(
            optimizer=optimizer,
            T_0=T_0,
            T_mult=T_mult,
            eta_min=eta_min
        )
    
    # training
    logger.info("Starting training...")
    best_val_loss = [float('inf')]  # Use list to allow modification in nested function
    global_step = 0
    last_val_loss = None
    epochs_without_improvement = 0
    
    # Track checkpoint paths for cleanup
    checkpoint_paths = deque(maxlen=keep_last_n)
    
    for epoch in range(num_epochs):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        logger.info(f"{'='*60}")
        
        # Prepare callbacks for step-based validation and checkpointing
        validate_callback = validate_epoch if validation_frequency == 'step' and every_n_steps is not None else None
        save_checkpoint_callback = save_checkpoint if save_every_n_steps is not None else None
        
        # train with optional step-based validation and checkpointing
        train_loss, global_step = train_epoch(
            model, train_loader, optimizer, device, global_step,
            val_loader=val_loader if validation_frequency == 'step' else None,
            validate_func=validate_callback,
            scheduler=scheduler,
            validation_frequency=validation_frequency,
            every_n_steps=every_n_steps,
            save_every_n_steps=save_every_n_steps,
            checkpoint_dir=checkpoint_dir,
            epoch_num=epoch,
            save_checkpoint_func=save_checkpoint_callback,
            best_val_loss=best_val_loss if save_best else None,
            save_best=save_best
        )
        
        # Determine if we should validate by epoch
        should_validate = False
        if validation_frequency == 'epoch':
            if (epoch + 1) % every_n_epochs == 0:
                should_validate = True
        
        # Validate by epoch if needed (default: validate every epoch for best practice with ViT-MAE)
        if should_validate or (validation_frequency == 'epoch' and (epoch + 1) % every_n_epochs == 0):
            val_loss = validate_epoch(model, val_loader, device)
            last_val_loss = val_loss
            
            # Save best model
            if save_best and val_loss < best_val_loss[0] - min_delta:
                best_val_loss[0] = val_loss
                epochs_without_improvement = 0
                best_model_path = checkpoint_dir / "vit_mae_best.pth"
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"✓ Best model saved (val_loss: {val_loss:.4f})")
            elif patience:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    logger.info(f"\nEarly stopping triggered after {epoch + 1} epochs (no improvement for {patience} epochs)")
                    break
        else:
            val_loss = last_val_loss if last_val_loss is not None else best_val_loss[0] if best_val_loss[0] != float('inf') else float('inf')
        
        # update learning rate
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log epoch summary
        logger.info(f"\nEpoch {epoch + 1}/{num_epochs} Summary:")
        logger.info(f"  Train Loss: {train_loss:.4f}")
        logger.info(f"  Val Loss:   {val_loss:.4f}")
        logger.info(f"  Learning Rate: {current_lr:.6f}")
        logger.info(f"  Global Step: {global_step}")
        if patience:
            logger.info(f"  Epochs without improvement: {epochs_without_improvement}/{patience}")
        
        # Save checkpoint by epoch
        if (epoch + 1) % save_every_n_epochs == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}_step_{global_step}.pth"
            save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, train_loss, val_loss, checkpoint_path)
            checkpoint_paths.append(checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")
            
            # Cleanup old checkpoints
            cleanup_old_checkpoints(checkpoint_dir, keep_last_n, prefix='checkpoint_')
    
    logger.info("\nTraining completed!")
    # save final model
    final_model_path = checkpoint_dir / "vit_mae_final.pth"
    torch.save(model.state_dict(), final_model_path)
    logger.info(f"Final model saved: {final_model_path}")

if __name__ == "__main__":
    main()
