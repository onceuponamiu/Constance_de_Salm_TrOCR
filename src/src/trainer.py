"""
TrOCR trainer for Constance de Salm HTR fine-tuning.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
import wandb
from transformers import Trainer, TrainingArguments
from transformers.trainer_callback import TrainerCallback
import numpy as np

from .model import TrOCRForCdS, create_training_args, setup_callbacks
from .data_loader import CdSDataLoader, collate_fn
from .evaluator import HTREvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HTRTrainerCallback(TrainerCallback):
    """Custom callback for HTR training with detailed logging."""
    
    def __init__(self, evaluator: HTREvaluator, log_predictions: bool = True):
        self.evaluator = evaluator
        self.log_predictions = log_predictions
        self.best_cer = float('inf')
        self.start_time = time.time()
    
    def on_evaluate(self, args, state, control, model, tokenizer, eval_dataloader, **kwargs):
        """Perform detailed evaluation after each eval step."""
        logger.info("Running detailed HTR evaluation...")
        
        # Get predictions for evaluation
        predictions = []
        references = []
        
        model.eval()
        with torch.no_grad():
            for batch in eval_dataloader:
                pixel_values = batch['pixel_values'].to(model.device)
                generated_ids = model.generate(pixel_values, max_length=384)
                
                batch_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                batch_references = batch['texts']
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # Calculate metrics
        metrics = self.evaluator.compute_metrics(predictions, references)
        
        # Log metrics
        for metric_name, value in metrics.items():
            logger.info(f"Eval {metric_name}: {value:.4f}")
        
        # Track best CER
        current_cer = metrics.get('cer', float('inf'))
        if current_cer < self.best_cer:
            self.best_cer = current_cer
            logger.info(f"New best CER: {current_cer:.4f}")
        
        # Log sample predictions
        if self.log_predictions and len(predictions) > 0:
            for i in range(min(3, len(predictions))):
                logger.info(f"Sample {i+1}:")
                logger.info(f"  Predicted: {predictions[i]}")
                logger.info(f"  Reference: {references[i]}")
                logger.info(f"  CER: {self.evaluator.compute_cer(predictions[i], references[i]):.4f}")
        
        return metrics
    
    def on_log(self, args, state, control, logs, **kwargs):
        """Log training progress."""
        if 'train_loss' in logs:
            elapsed_time = time.time() - self.start_time
            logger.info(f"Step {state.global_step}: Loss={logs['train_loss']:.4f}, "
                       f"Time={elapsed_time:.1f}s, Best CER={self.best_cer:.4f}")


class TrOCRTrainer:
    """Main trainer class for TrOCR fine-tuning on Constance de Salm data."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Initialize components
        self.model = None
        self.data_loader = None
        self.evaluator = None
        self.trainer = None
        
        # Setup directories
        self._setup_directories()
        
        # Setup logging
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories."""
        paths = self.config['paths']
        for path_key, path_value in paths.items():
            os.makedirs(path_value, exist_ok=True)
    
    def _setup_logging(self):
        """Setup experiment tracking."""
        experiment_config = self.config.get('experiment', {})
        
        # Weights & Biases setup
        if experiment_config.get('wandb', {}).get('enabled', False):
            wandb_config = experiment_config['wandb']
            wandb.init(
                project=wandb_config.get('project', 'trocr-htr'),
                entity=wandb_config.get('entity'),
                name=experiment_config.get('name', 'trocr_training'),
                tags=experiment_config.get('tags', []),
                notes=experiment_config.get('notes', ''),
                config=self.config
            )
            logger.info("Weights & Biases logging enabled")
    
    def initialize_model(self):
        """Initialize TrOCR model."""
        model_name = self.config['model']['name']
        self.model = TrOCRForCdS(model_name=model_name, config=self.config)
        self.model.to(self.device)
        
        # Log model info
        model_info = self.model.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
    
    def initialize_data(self):
        """Initialize data loaders."""
        self.data_loader = CdSDataLoader(self.config)
        self.train_loader, self.val_loader, self.test_loader = self.data_loader.create_dataloaders()
        
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")
        logger.info(f"Test samples: {len(self.test_loader.dataset)}")
        
        # Save dataset info
        self.data_loader.save_dataset_info(self.config['paths']['outputs_dir'])
    
    def initialize_evaluator(self):
        """Initialize evaluator."""
        self.evaluator = HTREvaluator(self.config['evaluation'])
    
    def create_trainer(self) -> Trainer:
        """Create HuggingFace Trainer instance."""
        # Training arguments
        training_args = create_training_args(
            self.config, 
            self.config['paths']['models_dir']
        )
        
        # Callbacks
        callbacks = setup_callbacks(self.config)
        callbacks.append(HTRTrainerCallback(self.evaluator))
        
        # Create trainer
        trainer = Trainer(
            model=self.model.model,  # Use the underlying transformers model
            args=training_args,
            train_dataset=self.train_loader.dataset,
            eval_dataset=self.val_loader.dataset,
            tokenizer=self.model.processor.tokenizer,
            data_collator=collate_fn,
            callbacks=callbacks,
        )
        
        return trainer
    
    def train(self):
        """Main training loop."""
        logger.info("Starting TrOCR training...")
        
        # Initialize all components
        self.initialize_model()
        self.initialize_data()
        self.initialize_evaluator()
        
        # Create trainer
        self.trainer = self.create_trainer()
        
        # Start training
        try:
            train_result = self.trainer.train()
            
            # Log training results
            logger.info("Training completed!")
            logger.info(f"Final train loss: {train_result.training_loss:.4f}")
            
            # Save final model
            final_model_path = Path(self.config['paths']['models_dir']) / "final"
            self.model.save_model(str(final_model_path))
            logger.info(f"Final model saved to {final_model_path}")
            
            return train_result
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            raise
        
        finally:
            # Cleanup
            if wandb.run is not None:
                wandb.finish()
    
    def evaluate_on_test(self) -> Dict:
        """Evaluate the trained model on test set."""
        if self.model is None or self.test_loader is None:
            raise ValueError("Model and data must be initialized before evaluation")
        
        logger.info("Evaluating on test set...")
        
        predictions = []
        references = []
        
        self.model.eval()
        with torch.no_grad():
            for batch in self.test_loader:
                pixel_values = batch['pixel_values'].to(self.device)
                generated_ids = self.model.generate(pixel_values, max_length=384)
                
                batch_predictions = self.model.processor.batch_decode(
                    generated_ids, skip_special_tokens=True
                )
                batch_references = batch['texts']
                
                predictions.extend(batch_predictions)
                references.extend(batch_references)
        
        # Compute comprehensive metrics
        test_metrics = self.evaluator.compute_metrics(predictions, references)
        
        # Log results
        logger.info("Test Results:")
        for metric_name, value in test_metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        # Save detailed results
        results_path = Path(self.config['paths']['outputs_dir']) / "test_evaluation.json"
        import json
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump({
                'metrics': test_metrics,
                'samples': list(zip(predictions[:10], references[:10]))  # Save first 10 samples
            }, f, indent=2, ensure_ascii=False)
        
        return test_metrics
    
    def resume_training(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming training from {checkpoint_path}")
        
        # Initialize components
        self.initialize_model()
        self.initialize_data()
        self.initialize_evaluator()
        
        # Create trainer
        self.trainer = self.create_trainer()
        
        # Resume training
        train_result = self.trainer.train(resume_from_checkpoint=checkpoint_path)
        
        logger.info("Training resumed and completed!")
        return train_result
    
    def save_checkpoint(self, checkpoint_name: str = "checkpoint"):
        """Save training checkpoint."""
        if self.trainer is None:
            raise ValueError("Trainer must be initialized before saving checkpoint")
        
        checkpoint_path = Path(self.config['paths']['models_dir']) / checkpoint_name
        self.trainer.save_model(str(checkpoint_path))
        logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        return str(checkpoint_path)


def quick_train_test(config: Dict):
    """Quick training test with minimal data for debugging."""
    # Modify config for quick test
    debug_config = config.copy()
    debug_config['training']['num_epochs'] = 1
    debug_config['training']['logging_steps'] = 10
    debug_config['debug'] = {'fast_dev_run': True}
    
    trainer = TrOCRTrainer(debug_config)
    trainer.train()
    
    return trainer


if __name__ == "__main__":
    # Test training setup
    import yaml
    
    # Load config
    with open('../config/training_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    # Load base config
    with open('../config/base_config.yaml', 'r') as f:
        base_config = yaml.safe_load(f)
    
    # Merge configs
    config = {**base_config, **config}
    
    # Run quick test
    trainer = quick_train_test(config)
    print("Training test completed successfully!")
