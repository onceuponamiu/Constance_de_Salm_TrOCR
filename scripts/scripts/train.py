#!/usr/bin/env python3
"""
Main training script for TrOCR fine-tuning on Constance de Salm data.
"""

import argparse
import logging
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.trainer import TrOCRTrainer
from src.utils import load_config, setup_logging, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Train TrOCR model on Constance de Salm handwriting data"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/training_config.yaml",
        help="Path to training configuration file"
    )
    parser.add_argument(
        "--base_config", 
        type=str, 
        default="config/base_config.yaml",
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--resume_from_checkpoint", 
        type=str,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--output_dir", 
        type=str,
        help="Override output directory for models"
    )
    parser.add_argument(
        "--experiment_name", 
        type=str,
        help="Override experiment name"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    parser.add_argument(
        "--fast_dev_run", 
        action="store_true",
        help="Run a fast development test with minimal data"
    )
    parser.add_argument(
        "--no_wandb", 
        action="store_true",
        help="Disable Weights & Biases logging"
    )
    parser.add_argument(
        "--gpu_id", 
        type=int,
        help="Specific GPU ID to use"
    )
    parser.add_argument(
        "--eval_only", 
        action="store_true",
        help="Only run evaluation on test set (requires trained model)"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"training_{Path(args.config).stem}.log"
    setup_logging(args.log_level, str(log_file))
    
    # Load configuration
    try:
        config = load_config(args.config, args.base_config)
        logger.info(f"Loaded configuration from {args.config}")
        if args.base_config:
            logger.info(f"Base configuration from {args.base_config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Apply command line overrides
    if args.output_dir:
        config['paths']['models_dir'] = args.output_dir
        logger.info(f"Output directory overridden: {args.output_dir}")
    
    if args.experiment_name:
        config['experiment']['name'] = args.experiment_name
        logger.info(f"Experiment name overridden: {args.experiment_name}")
    
    if args.fast_dev_run:
        config['debug']['fast_dev_run'] = True
        config['training']['num_epochs'] = 1
        config['training']['logging_steps'] = 5
        config['training']['eval_steps'] = 10
        logger.info("Fast development run enabled")
    
    if args.no_wandb:
        config['experiment']['wandb']['enabled'] = False
        logger.info("Weights & Biases logging disabled")
    
    if args.gpu_id is not None:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
        logger.info(f"Using GPU ID: {args.gpu_id}")
    
    # Check device availability
    device = get_device()
    logger.info(f"Training will use device: {device}")
    
    # Check if we have training data
    data_config = config['data']
    source_dirs = data_config['source_dirs']
    
    # Convert relative paths to absolute paths
    script_dir = Path(__file__).parent.parent
    for i, source_dir in enumerate(source_dirs):
        if not Path(source_dir).is_absolute():
            source_dirs[i] = str(script_dir / source_dir)
    
    logger.info(f"Source directories: {source_dirs}")
    
    # Verify source directories exist
    missing_dirs = []
    for source_dir in source_dirs:
        if not Path(source_dir).exists():
            missing_dirs.append(source_dir)
    
    if missing_dirs:
        logger.error(f"Source directories not found: {missing_dirs}")
        logger.error("Please run prepare_data.py first or check directory paths")
        return 1
    
    try:
        # Initialize trainer
        logger.info("Initializing TrOCR trainer...")
        trainer = TrOCRTrainer(config)
        
        if args.eval_only:
            # Only run evaluation
            logger.info("Running evaluation only...")
            
            # Check if model exists
            model_path = Path(config['paths']['models_dir']) / "final"
            if not model_path.exists():
                logger.error(f"No trained model found at {model_path}")
                logger.error("Please train a model first")
                return 1
            
            # Load model and evaluate
            trainer.initialize_model()
            trainer.initialize_data()
            trainer.initialize_evaluator()
            
            # Load trained weights
            import torch
            model_weights = torch.load(model_path / "pytorch_model.bin", map_location=trainer.device)
            trainer.model.model.load_state_dict(model_weights)
            
            # Evaluate on test set
            test_metrics = trainer.evaluate_on_test()
            
            logger.info("Evaluation completed!")
            for metric, value in test_metrics.items():
                logger.info(f"{metric}: {value:.4f}")
            
        elif args.resume_from_checkpoint:
            # Resume training from checkpoint
            logger.info(f"Resuming training from checkpoint: {args.resume_from_checkpoint}")
            
            if not Path(args.resume_from_checkpoint).exists():
                logger.error(f"Checkpoint not found: {args.resume_from_checkpoint}")
                return 1
            
            train_result = trainer.resume_training(args.resume_from_checkpoint)
            logger.info("Training resumed and completed!")
            
        else:
            # Start fresh training
            logger.info("Starting training from scratch...")
            train_result = trainer.train()
            
            # Evaluate on test set after training
            logger.info("Evaluating on test set...")
            test_metrics = trainer.evaluate_on_test()
            
            logger.info("Training and evaluation completed!")
            logger.info("Final test metrics:")
            for metric, value in test_metrics.items():
                logger.info(f"  {metric}: {value:.4f}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        
        # Save checkpoint if trainer exists
        if 'trainer' in locals() and trainer.trainer is not None:
            checkpoint_path = trainer.save_checkpoint("interrupted_checkpoint")
            logger.info(f"Checkpoint saved to: {checkpoint_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
