"""
Data loader for TrOCR fine-tuning on Constance de Salm correspondence data.
Handles image preprocessing and text extraction from XML files.
"""

import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import logging

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import TrOCRProcessor
import albumentations as A
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CdSDataset(Dataset):
    """Dataset class for Constance de Salm HTR data."""
    
    def __init__(
        self, 
        image_paths: List[str], 
        texts: List[str], 
        processor: TrOCRProcessor,
        augmentation: Optional[A.Compose] = None,
        max_length: int = 384
    ):
        self.image_paths = image_paths
        self.texts = texts
        self.processor = processor
        self.augmentation = augmentation
        self.max_length = max_length
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Load and preprocess image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert('RGB')
        
        # Apply augmentation if specified
        if self.augmentation:
            image_array = np.array(image)
            augmented = self.augmentation(image=image_array)
            image = Image.fromarray(augmented['image'])
        
        # Process with TrOCR processor
        text = self.texts[idx]
        
        # Encode image and text
        pixel_values = self.processor(image, return_tensors="pt").pixel_values.squeeze()
        labels = self.processor.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).input_ids.squeeze()
        
        return {
            'pixel_values': pixel_values,
            'labels': labels,
            'text': text,
            'image_path': image_path
        }


class CdSDataLoader:
    """Data loader for preparing TrOCR training data from Constance de Salm corpus."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.processor = TrOCRProcessor.from_pretrained(config['model']['name'])
        
        # Setup augmentation
        self.augmentation = self._setup_augmentation() if config['data'].get('augmentation', {}).get('enabled', False) else None
        
    def _setup_augmentation(self) -> A.Compose:
        """Setup image augmentation pipeline."""
        aug_config = self.config['data']['augmentation']
        
        transforms = []
        
        # Rotation
        if 'rotation_range' in aug_config:
            rot_range = aug_config['rotation_range']
            transforms.append(A.Rotate(limit=rot_range, p=0.3))
        
        # Brightness and contrast
        if 'brightness_range' in aug_config:
            br_range = aug_config['brightness_range']
            transforms.append(A.RandomBrightnessContrast(
                brightness_limit=br_range[1] - 1.0,
                contrast_limit=aug_config.get('contrast_range', [0.8, 1.2])[1] - 1.0,
                p=0.3
            ))
        
        # Noise
        if 'noise_factor' in aug_config:
            transforms.append(A.GaussNoise(var_limit=(0, aug_config['noise_factor']), p=0.2))
        
        # Blur
        if aug_config.get('blur_probability', 0) > 0:
            transforms.append(A.Blur(blur_limit=3, p=aug_config['blur_probability']))
        
        return A.Compose(transforms)
    
    def extract_text_from_xml(self, xml_path: str) -> str:
        """
        Extract text from ALTO XML files.
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Extracted text string
        """
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            # Handle different XML namespaces
            namespaces = {
                'alto': 'http://www.loc.gov/standards/alto/ns-v4#',
                'tei': 'http://www.tei-c.org/ns/1.0'
            }
            
            texts = []
            
            # Try ALTO format first
            for ns_prefix, ns_url in namespaces.items():
                # Look for String elements (ALTO) or text elements (TEI)
                string_elements = root.findall(f'.//{{{ns_url}}}String') or root.findall(f'.//{{{ns_url}}}text')
                
                if string_elements:
                    for elem in string_elements:
                        content = elem.get('CONTENT') or elem.text
                        if content:
                            texts.append(content.strip())
                    break
            
            # If no namespaced elements found, try without namespace
            if not texts:
                for elem in root.iter():
                    if elem.tag.lower() in ['string', 'text', 'line']:
                        content = elem.get('CONTENT') or elem.text
                        if content:
                            texts.append(content.strip())
            
            return ' '.join(texts) if texts else ""
            
        except Exception as e:
            logger.warning(f"Error parsing XML file {xml_path}: {e}")
            return ""
    
    def preprocess_text(self, text: str) -> str:
        """
        Preprocess text according to configuration.
        
        Args:
            text: Raw text string
            
        Returns:
            Preprocessed text
        """
        text_config = self.config['data'].get('text', {})
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Preserve historical characteristics unless specified otherwise
        if not text_config.get('preserve_accents', True):
            # Remove accents (not recommended for historical texts)
            import unicodedata
            text = unicodedata.normalize('NFD', text)
            text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
        
        # Handle abbreviations (expand common ones)
        if text_config.get('handle_abbreviations', False):
            abbreviations = {
                'M.': 'Monsieur',
                'Mme': 'Madame',
                'etc.': 'et cetera',
                # Add more as needed
            }
            for abbr, expansion in abbreviations.items():
                text = text.replace(abbr, expansion)
        
        return text
    
    def find_image_text_pairs(self, source_dirs: List[str]) -> Tuple[List[str], List[str]]:
        """
        Find matching image and text file pairs from source directories.
        
        Args:
            source_dirs: List of source directories to search
            
        Returns:
            Tuple of (image_paths, texts)
        """
        image_paths = []
        texts = []
        
        for source_dir in source_dirs:
            source_path = Path(source_dir)
            if not source_path.exists():
                logger.warning(f"Source directory does not exist: {source_dir}")
                continue
                
            logger.info(f"Processing directory: {source_dir}")
            
            # Find all image files
            image_extensions = {'.jpg', '.jpeg', '.png', '.tif', '.tiff'}
            image_files = []
            
            for ext in image_extensions:
                image_files.extend(source_path.glob(f"**/*{ext}"))
                image_files.extend(source_path.glob(f"**/*{ext.upper()}"))
            
            for image_file in image_files:
                # Look for corresponding XML file
                xml_candidates = [
                    image_file.with_suffix('.xml'),
                    image_file.parent / (image_file.stem + '.xml')
                ]
                
                xml_file = None
                for candidate in xml_candidates:
                    if candidate.exists():
                        xml_file = candidate
                        break
                
                if xml_file:
                    # Extract text from XML
                    text = self.extract_text_from_xml(str(xml_file))
                    text = self.preprocess_text(text)
                    
                    if text:  # Only include pairs with non-empty text
                        image_paths.append(str(image_file))
                        texts.append(text)
                        logger.debug(f"Added pair: {image_file.name} -> {len(text)} chars")
                else:
                    logger.debug(f"No XML found for image: {image_file}")
        
        logger.info(f"Found {len(image_paths)} image-text pairs")
        return image_paths, texts
    
    def create_datasets(self) -> Tuple[Dataset, Dataset, Dataset]:
        """
        Create train, validation, and test datasets.
        
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Find all image-text pairs
        source_dirs = self.config['data']['source_dirs']
        image_paths, texts = self.find_image_text_pairs(source_dirs)
        
        if len(image_paths) == 0:
            raise ValueError("No image-text pairs found in source directories")
        
        # Split data
        train_split = self.config['data']['train_split']
        val_split = self.config['data']['val_split']
        
        # First split: train + val vs test
        train_val_images, test_images, train_val_texts, test_texts = train_test_split(
            image_paths, texts, 
            test_size=1 - train_split - val_split,
            random_state=42,
            stratify=None
        )
        
        # Second split: train vs val
        val_ratio = val_split / (train_split + val_split)
        train_images, val_images, train_texts, val_texts = train_test_split(
            train_val_images, train_val_texts,
            test_size=val_ratio,
            random_state=42,
            stratify=None
        )
        
        logger.info(f"Dataset splits - Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")
        
        # Create datasets
        max_length = self.config['model']['max_length']
        
        train_dataset = CdSDataset(
            train_images, train_texts, self.processor, 
            augmentation=self.augmentation, max_length=max_length
        )
        
        val_dataset = CdSDataset(
            val_images, val_texts, self.processor, 
            augmentation=None, max_length=max_length
        )
        
        test_dataset = CdSDataset(
            test_images, test_texts, self.processor, 
            augmentation=None, max_length=max_length
        )
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """
        Create PyTorch DataLoaders for training.
        
        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        batch_size = self.config['training']['batch_size']
        num_workers = self.config['hardware']['dataloader_num_workers']
        pin_memory = self.config['hardware']['pin_memory']
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=self.config['training'].get('dataloader_drop_last', False)
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        
        return train_loader, val_loader, test_loader
    
    def save_dataset_info(self, output_dir: str):
        """
        Save dataset information for later reference.
        
        Args:
            output_dir: Directory to save dataset info
        """
        train_dataset, val_dataset, test_dataset = self.create_datasets()
        
        # Create dataset info
        info = {
            'train_size': len(train_dataset),
            'val_size': len(val_dataset),
            'test_size': len(test_dataset),
            'total_size': len(train_dataset) + len(val_dataset) + len(test_dataset),
            'config': self.config
        }
        
        # Save train/val/test file lists
        datasets = {
            'train': train_dataset,
            'val': val_dataset,
            'test': test_dataset
        }
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for split_name, dataset in datasets.items():
            df = pd.DataFrame({
                'image_path': dataset.image_paths,
                'text': dataset.texts,
                'text_length': [len(text) for text in dataset.texts]
            })
            df.to_csv(output_path / f"{split_name}_dataset.csv", index=False)
            
        # Save overall info
        import json
        with open(output_path / "dataset_info.json", 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Dataset info saved to {output_dir}")


def collate_fn(batch):
    """Custom collate function for TrOCR data."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    
    return {
        'pixel_values': pixel_values,
        'labels': labels,
        'texts': [item['text'] for item in batch],
        'image_paths': [item['image_path'] for item in batch]
    }


if __name__ == "__main__":
    # Test data loader
    import yaml
    
    # Load config
    with open('../config/base_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Create data loader
    data_loader = CdSDataLoader(config)
    
    # Test finding image-text pairs
    image_paths, texts = data_loader.find_image_text_pairs(config['data']['source_dirs'])
    print(f"Found {len(image_paths)} image-text pairs")
    
    if image_paths:
        print(f"Sample text: {texts[0][:100]}...")
