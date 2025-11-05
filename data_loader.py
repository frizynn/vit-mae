import torch
from torch.utils.data import Dataset, DataLoader, random_split
from pathlib import Path
import yaml
import pandas as pd
from PIL import Image
import numpy as np
import random
import logging

logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    """Dataset for loading images from file paths"""
    def __init__(self, image_paths, img_size=224, transform=None):
        self.image_paths = image_paths
        self.img_size = img_size
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
            img = img.resize((self.img_size, self.img_size))
            
            # convert to tensor and normalize to [0, 1]
            img_array = np.array(img).astype(np.float32) / 255.0
            img_tensor = torch.from_numpy(img_array).permute(2, 0, 1)  # HWC -> CHW
            
            # normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img_tensor = (img_tensor - mean) / std
            
            return img_tensor
        except Exception as e:
            logger.error(f"Error loading {img_path}: {e}")
            # return a black image as fallback
            return torch.zeros(3, self.img_size, self.img_size)

def load_flat_dataset(dataset_path, split_ratio=0.8, seed=42):
    """Load dataset with flat structure (images directly in path)"""
    dataset_path = Path(dataset_path)
    image_paths = []
    
    # find all image files directly in the path
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(list(dataset_path.glob(ext)))
    
    image_paths = sorted(image_paths)
    logger.info(f"  Found {len(image_paths)} images in flat structure")
    
    # split train/val
    random.seed(seed)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * split_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    return train_paths, val_paths

def load_nested_dataset(dataset_path, split_ratio=0.8, seed=42):
    """Load dataset with nested structure (images in subdirectories)"""
    dataset_path = Path(dataset_path)
    image_paths = []
    
    # find all image files recursively
    for ext in ['*.png', '*.jpg', '*.jpeg']:
        image_paths.extend(list(dataset_path.rglob(ext)))
    
    image_paths = sorted(image_paths)
    logger.info(f"  Found {len(image_paths)} images in nested structure")
    
    # split train/val
    random.seed(seed)
    random.shuffle(image_paths)
    split_idx = int(len(image_paths) * split_ratio)
    train_paths = image_paths[:split_idx]
    val_paths = image_paths[split_idx:]
    
    return train_paths, val_paths

def load_csv_dataset(dataset_path, annotations_path, seed=42):
    """Load dataset using CSV files for train/val split"""
    dataset_path = Path(dataset_path)
    annotations_path = Path(annotations_path)
    
    train_csv = annotations_path / "train_final.csv"
    val_csv = annotations_path / "val_final.csv"
    
    train_paths = []
    val_paths = []
    
    # load train images
    if train_csv.exists():
        df_train = pd.read_csv(train_csv)
        for _, row in df_train.iterrows():
            img_path = dataset_path / row['FullFile']
            if img_path.exists():
                train_paths.append(img_path)
        logger.info(f"  Found {len(train_paths)} train images from CSV")
    else:
        logger.warning(f"  Warning: {train_csv} not found")
    
    # load val images
    if val_csv.exists():
        df_val = pd.read_csv(val_csv)
        for _, row in df_val.iterrows():
            img_path = dataset_path / row['FullFile']
            if img_path.exists():
                val_paths.append(img_path)
        logger.info(f"  Found {len(val_paths)} val images from CSV")
    else:
        logger.warning(f"  Warning: {val_csv} not found")
    
    return train_paths, val_paths

def load_datasets_from_config(config_path="config.yaml", img_size=224, batch_size=8, num_workers=2, seed=42):
    """Load all datasets from config.yaml and create train/val dataloaders"""
    
    # load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # get seed from config if not provided
    if seed is None:
        seed = config.get('training', {}).get('seed', 42)
    
    all_train_paths = []
    all_val_paths = []
    
    # load each dataset
    for dataset_config in config['datasets']:
        name = dataset_config['name']
        path = dataset_config['path']
        structure = dataset_config.get('structure', 'flat')
        
        logger.info(f"Loading dataset: {name}")
        logger.info(f"  Path: {path}")
        logger.info(f"  Structure: {structure}")
        
        if structure == 'flat':
            train_paths, val_paths = load_flat_dataset(
                path, 
                split_ratio=dataset_config.get('split_ratio', 0.8),
                seed=seed
            )
        elif structure == 'nested':
            train_paths, val_paths = load_nested_dataset(
                path,
                split_ratio=dataset_config.get('split_ratio', 0.8),
                seed=seed
            )
        elif structure == 'csv':
            annotations_path = dataset_config.get('annotations_path', '')
            train_paths, val_paths = load_csv_dataset(path, annotations_path, seed=seed)
        else:
            logger.warning(f"  Unknown structure '{structure}', skipping")
            continue
        
        all_train_paths.extend(train_paths)
        all_val_paths.extend(val_paths)
        logger.info(f"  Added {len(train_paths)} train, {len(val_paths)} val images")
    
    logger.info(f"Total: {len(all_train_paths)} train images, {len(all_val_paths)} val images")
    
    # create datasets
    train_dataset = ImageDataset(all_train_paths, img_size=img_size)
    val_dataset = ImageDataset(all_val_paths, img_size=img_size)
    
    # create generator for reproducibility
    generator = torch.Generator()
    generator.manual_seed(seed)
    
    # create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=generator
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

