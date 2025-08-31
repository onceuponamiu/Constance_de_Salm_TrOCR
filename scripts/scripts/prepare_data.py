#!/usr/bin/env python3
"""
Script to prepare data for TrOCR training from Constance de Salm corpus.
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.data_loader import CdSDataLoader
from src.utils import load_config, setup_logging, save_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare TrOCR training data from Constance de Salm corpus"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--source_dirs", 
        nargs="+",
        help="Source directories containing images and XML files"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--dry_run", 
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override source directories if provided
    if args.source_dirs:
        config['data']['source_dirs'] = args.source_dirs
        logger.info(f"Using source directories: {args.source_dirs}")
    
    # Override output directory if provided
    if args.output_dir != "data/":
        config['data']['output_dir'] = args.output_dir
        logger.info(f"Using output directory: {args.output_dir}")
    
    try:
        # Initialize data loader
        data_loader = CdSDataLoader(config)
        
        # Find image-text pairs
        logger.info("Scanning source directories for image-text pairs...")
        image_paths, texts = data_loader.find_image_text_pairs(config['data']['source_dirs'])
        
        if not image_paths:
            logger.error("No image-text pairs found in source directories")
            return 1
        
        logger.info(f"Found {len(image_paths)} image-text pairs")
        
        # Print statistics
        text_lengths = [len(text) for text in texts]
        logger.info(f"Text length statistics:")
        logger.info(f"  Min: {min(text_lengths)} characters")
        logger.info(f"  Max: {max(text_lengths)} characters")
        logger.info(f"  Average: {sum(text_lengths) / len(text_lengths):.1f} characters")
        
        # Show sample data
        logger.info("Sample data:")
        for i in range(min(3, len(image_paths))):
            logger.info(f"  Image: {Path(image_paths[i]).name}")
            logger.info(f"  Text: {texts[i][:100]}{'...' if len(texts[i]) > 100 else ''}")
            logger.info("")
        
        if args.dry_run:
            logger.info("Dry run completed. No data was saved.")
            return 0
        
        # Create datasets and save information
        logger.info("Creating train/validation/test splits...")
        train_dataset, val_dataset, test_dataset = data_loader.create_datasets()
        
        logger.info(f"Dataset splits:")
        logger.info(f"  Training: {len(train_dataset)} samples")
        logger.info(f"  Validation: {len(val_dataset)} samples")
        logger.info(f"  Test: {len(test_dataset)} samples")
        
        # Save dataset information
        output_dir = Path(args.output_dir)
        data_loader.save_dataset_info(str(output_dir))
        
        # Save configuration used
        config_output_path = output_dir / "config_used.yaml"
        import yaml
        with open(config_output_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False)
        
        logger.info(f"Data preparation completed successfully!")
        logger.info(f"Output saved to: {output_dir}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Data preparation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
