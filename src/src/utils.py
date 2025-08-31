"""
Utility functions for TrOCR-HTR project.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import torch
import numpy as np
from PIL import Image
import cv2

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_config(config_path: str, base_config_path: Optional[str] = None) -> Dict:
    """
    Load configuration from YAML file with optional base config.
    
    Args:
        config_path: Path to main config file
        base_config_path: Path to base config file (optional)
        
    Returns:
        Merged configuration dictionary
    """
    config = {}
    
    # Load base config if provided
    if base_config_path and os.path.exists(base_config_path):
        with open(base_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        logger.info(f"Loaded base config from {base_config_path}")
    
    # Load main config
    if os.path.exists(config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            main_config = yaml.safe_load(f)
        
        # Merge configs (main config overrides base config)
        config = deep_merge_dicts(config, main_config)
        logger.info(f"Loaded config from {config_path}")
    else:
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    return config


def deep_merge_dicts(base_dict: Dict, override_dict: Dict) -> Dict:
    """
    Deep merge two dictionaries.
    
    Args:
        base_dict: Base dictionary
        override_dict: Dictionary with override values
        
    Returns:
        Merged dictionary
    """
    result = base_dict.copy()
    
    for key, value in override_dict.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge_dicts(result[key], value)
        else:
            result[key] = value
    
    return result


def save_json(data: Any, filepath: str, indent: int = 2):
    """
    Save data to JSON file.
    
    Args:
        data: Data to save
        filepath: Output file path
        indent: JSON indentation
    """
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=indent, ensure_ascii=False)
    
    logger.info(f"Data saved to {filepath}")


def load_json(filepath: str) -> Any:
    """
    Load data from JSON file.
    
    Args:
        filepath: Input file path
        
    Returns:
        Loaded data
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Data loaded from {filepath}")
    return data


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional log file path
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)
    root_logger.addHandler(console_handler)
    
    # Setup file handler if log_file is provided
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
        
        logger.info(f"Logging to file: {log_file}")


def get_device(prefer_gpu: bool = True) -> torch.device:
    """
    Get the best available device for training/inference.
    
    Args:
        prefer_gpu: Whether to prefer GPU over CPU
        
    Returns:
        PyTorch device
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    
    return device


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    """
    Count model parameters.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'frozen': total_params - trainable_params
    }


def format_number(num: Union[int, float], precision: int = 2) -> str:
    """
    Format large numbers with appropriate units.
    
    Args:
        num: Number to format
        precision: Decimal precision
        
    Returns:
        Formatted string
    """
    if num >= 1e9:
        return f"{num/1e9:.{precision}f}B"
    elif num >= 1e6:
        return f"{num/1e6:.{precision}f}M"
    elif num >= 1e3:
        return f"{num/1e3:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def create_directory_structure(base_path: str, directories: List[str]):
    """
    Create directory structure.
    
    Args:
        base_path: Base directory path
        directories: List of subdirectories to create
    """
    base_path = Path(base_path)
    base_path.mkdir(parents=True, exist_ok=True)
    
    for directory in directories:
        dir_path = base_path / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Created directory: {dir_path}")
    
    logger.info(f"Directory structure created in {base_path}")


class ImageProcessor:
    """Utility class for image processing operations."""
    
    def __init__(self, target_size: tuple = (384, 384)):
        self.target_size = target_size
    
    def resize_image(self, image: Image.Image, maintain_aspect: bool = True) -> Image.Image:
        """
        Resize image to target size.
        
        Args:
            image: PIL Image
            maintain_aspect: Whether to maintain aspect ratio
            
        Returns:
            Resized image
        """
        if maintain_aspect:
            # Calculate aspect ratio preserving resize
            img_ratio = image.width / image.height
            target_ratio = self.target_size[0] / self.target_size[1]
            
            if img_ratio > target_ratio:
                # Image is wider than target
                new_width = self.target_size[0]
                new_height = int(new_width / img_ratio)
            else:
                # Image is taller than target
                new_height = self.target_size[1]
                new_width = int(new_height * img_ratio)
            
            # Resize and pad
            resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Create new image with target size and paste resized image
            new_image = Image.new('RGB', self.target_size, (255, 255, 255))
            paste_x = (self.target_size[0] - new_width) // 2
            paste_y = (self.target_size[1] - new_height) // 2
            new_image.paste(resized, (paste_x, paste_y))
            
            return new_image
        else:
            return image.resize(self.target_size, Image.Resampling.LANCZOS)
    
    def enhance_image(self, image: Image.Image, 
                     brightness: float = 1.0, 
                     contrast: float = 1.0,
                     sharpness: float = 1.0) -> Image.Image:
        """
        Enhance image with brightness, contrast, and sharpness adjustments.
        
        Args:
            image: PIL Image
            brightness: Brightness factor (1.0 = no change)
            contrast: Contrast factor (1.0 = no change)
            sharpness: Sharpness factor (1.0 = no change)
            
        Returns:
            Enhanced image
        """
        from PIL import ImageEnhance
        
        # Apply enhancements
        if brightness != 1.0:
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(brightness)
        
        if contrast != 1.0:
            enhancer = ImageEnhance.Contrast(image)
            image = enhancer.enhance(contrast)
        
        if sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(image)
            image = enhancer.enhance(sharpness)
        
        return image
    
    def preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """
        Preprocess image for optimal OCR performance.
        
        Args:
            image: PIL Image
            
        Returns:
            Preprocessed image
        """
        # Convert to grayscale and back to RGB for consistency
        gray = image.convert('L')
        
        # Apply slight contrast enhancement
        enhanced = self.enhance_image(
            gray.convert('RGB'), 
            contrast=1.1,
            sharpness=1.05
        )
        
        # Resize to target size
        resized = self.resize_image(enhanced)
        
        return resized


class MetricsTracker:
    """Utility class for tracking training metrics."""
    
    def __init__(self):
        self.metrics_history = {}
        self.best_metrics = {}
    
    def update(self, metrics: Dict[str, float], step: int):
        """
        Update metrics history.
        
        Args:
            metrics: Dictionary of metric values
            step: Training step number
        """
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_history:
                self.metrics_history[metric_name] = []
            
            self.metrics_history[metric_name].append((step, value))
            
            # Track best metrics (lower is better for most HTR metrics)
            if metric_name not in self.best_metrics:
                self.best_metrics[metric_name] = value
            elif metric_name in ['cer', 'wer', 'loss']:  # Lower is better
                if value < self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
            else:  # Higher is better
                if value > self.best_metrics[metric_name]:
                    self.best_metrics[metric_name] = value
    
    def get_best_metrics(self) -> Dict[str, float]:
        """Get best recorded metrics."""
        return self.best_metrics.copy()
    
    def save_history(self, filepath: str):
        """Save metrics history to file."""
        save_json({
            'history': self.metrics_history,
            'best': self.best_metrics
        }, filepath)


def estimate_training_time(num_samples: int, batch_size: int, num_epochs: int,
                          seconds_per_batch: float = 2.0) -> Dict[str, float]:
    """
    Estimate training time.
    
    Args:
        num_samples: Number of training samples
        batch_size: Batch size
        num_epochs: Number of epochs
        seconds_per_batch: Estimated seconds per batch
        
    Returns:
        Time estimates in different units
    """
    batches_per_epoch = num_samples // batch_size
    total_batches = batches_per_epoch * num_epochs
    total_seconds = total_batches * seconds_per_batch
    
    return {
        'total_seconds': total_seconds,
        'total_minutes': total_seconds / 60,
        'total_hours': total_seconds / 3600,
        'batches_per_epoch': batches_per_epoch,
        'total_batches': total_batches
    }


if __name__ == "__main__":
    # Test utilities
    
    # Test config loading
    try:
        config = load_config('../config/base_config.yaml')
        print("Config loaded successfully")
        print(f"Model name: {config.get('model', {}).get('name', 'Not found')}")
    except Exception as e:
        print(f"Config loading failed: {e}")
    
    # Test device detection
    device = get_device()
    print(f"Selected device: {device}")
    
    # Test number formatting
    test_numbers = [1234, 1234567, 1234567890]
    for num in test_numbers:
        print(f"{num} -> {format_number(num)}")
    
    # Test training time estimation
    time_est = estimate_training_time(1000, 8, 10)
    print(f"Estimated training time: {time_est['total_hours']:.1f} hours")
