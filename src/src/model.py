"""
TrOCR model wrapper for Constance de Salm HTR fine-tuning.
"""

import logging
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import torch
import torch.nn as nn
from transformers import (
    VisionEncoderDecoderModel,
    TrOCRProcessor,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback
)
from transformers.modeling_outputs import Seq2SeqLMOutput

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrOCRForCdS(nn.Module):
    """
    TrOCR model wrapper for Constance de Salm handwriting recognition.
    """
    
    def __init__(self, model_name: str = "microsoft/trocr-base-handwritten", config: Optional[Dict] = None):
        super().__init__()
        
        self.config = config or {}
        self.model_name = model_name
        
        # Load pre-trained TrOCR model
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.processor = TrOCRProcessor.from_pretrained(model_name)
        
        # Apply configuration
        self._configure_model()
        
        logger.info(f"Initialized TrOCR model: {model_name}")
        logger.info(f"Encoder: {self.model.encoder.__class__.__name__}")
        logger.info(f"Decoder: {self.model.decoder.__class__.__name__}")
    
    def _configure_model(self):
        """Configure model based on provided config."""
        model_config = self.config.get('model', {})
        
        # Set maximum sequence length
        if 'max_length' in model_config:
            max_length = model_config['max_length']
            self.model.config.max_length = max_length
            self.model.config.max_new_tokens = max_length
            
        # Freeze encoder if specified
        if model_config.get('freeze_encoder', False):
            for param in self.model.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen for training")
        
        # Enable gradient checkpointing for memory efficiency
        if model_config.get('gradient_checkpointing', False):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled")
    
    def forward(self, pixel_values: torch.Tensor, labels: Optional[torch.Tensor] = None) -> Seq2SeqLMOutput:
        """
        Forward pass for training.
        
        Args:
            pixel_values: Input images tensor
            labels: Target text token IDs (for training)
            
        Returns:
            Model output with loss (if labels provided)
        """
        return self.model(pixel_values=pixel_values, labels=labels)
    
    def generate(
        self, 
        pixel_values: torch.Tensor, 
        max_length: Optional[int] = None,
        num_beams: int = 5,
        early_stopping: bool = True,
        **kwargs
    ) -> torch.Tensor:
        """
        Generate text from images.
        
        Args:
            pixel_values: Input images tensor
            max_length: Maximum generation length
            num_beams: Number of beams for beam search
            early_stopping: Whether to stop early
            
        Returns:
            Generated token IDs
        """
        max_length = max_length or self.model.config.max_length
        
        return self.model.generate(
            pixel_values=pixel_values,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=early_stopping,
            **kwargs
        )
    
    def predict_text(self, images: Union[torch.Tensor, list], batch_size: int = 1) -> list:
        """
        Predict text from images with post-processing.
        
        Args:
            images: Input images (tensor or list of PIL images)
            batch_size: Batch size for processing
            
        Returns:
            List of predicted text strings
        """
        self.eval()
        predictions = []
        
        with torch.no_grad():
            if isinstance(images, list):
                # Process in batches
                for i in range(0, len(images), batch_size):
                    batch_images = images[i:i + batch_size]
                    
                    # Process images
                    pixel_values = self.processor(
                        batch_images, 
                        return_tensors="pt"
                    ).pixel_values
                    
                    if torch.cuda.is_available():
                        pixel_values = pixel_values.cuda()
                    
                    # Generate
                    generated_ids = self.generate(pixel_values)
                    
                    # Decode
                    batch_texts = self.processor.batch_decode(
                        generated_ids, 
                        skip_special_tokens=True
                    )
                    
                    predictions.extend(batch_texts)
            else:
                # Single tensor input
                generated_ids = self.generate(images)
                predictions = self.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
        
        return predictions
    
    def save_model(self, save_path: str):
        """
        Save model and processor to specified path.
        
        Args:
            save_path: Directory to save model
        """
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self.model.save_pretrained(save_path)
        self.processor.save_pretrained(save_path)
        
        # Save config
        import json
        with open(save_path / "model_config.json", 'w') as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Model saved to {save_path}")
    
    @classmethod
    def load_model(cls, model_path: str, config: Optional[Dict] = None):
        """
        Load a fine-tuned model from path.
        
        Args:
            model_path: Path to saved model
            config: Optional config override
            
        Returns:
            Loaded TrOCRForCdS instance
        """
        model_path = Path(model_path)
        
        # Load config if exists
        config_path = model_path / "model_config.json"
        if config_path.exists() and config is None:
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
        
        # Create instance
        instance = cls(model_name=str(model_path), config=config)
        
        logger.info(f"Model loaded from {model_path}")
        return instance
    
    def get_model_info(self) -> Dict:
        """Get model information and statistics."""
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            'model_name': self.model_name,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'frozen_parameters': total_params - trainable_params,
            'encoder_type': self.model.encoder.__class__.__name__,
            'decoder_type': self.model.decoder.__class__.__name__,
            'max_length': self.model.config.max_length,
            'vocab_size': self.model.config.decoder.vocab_size
        }


class TrOCRTrainerCallback:
    """Custom callback for TrOCR training with HTR-specific logging."""
    
    def __init__(self, log_frequency: int = 50):
        self.log_frequency = log_frequency
        self.step_count = 0
    
    def on_log(self, logs: Dict):
        """Called when logging occurs."""
        self.step_count += 1
        
        if self.step_count % self.log_frequency == 0:
            logger.info(f"Step {self.step_count}: {logs}")
    
    def on_evaluate(self, logs: Dict):
        """Called after evaluation."""
        logger.info(f"Evaluation results: {logs}")


def create_training_args(config: Dict, output_dir: str) -> TrainingArguments:
    """
    Create TrainingArguments from config.
    
    Args:
        config: Training configuration
        output_dir: Output directory for training
        
    Returns:
        TrainingArguments instance
    """
    training_config = config['training']
    hardware_config = config['hardware']
    
    args = TrainingArguments(
        output_dir=output_dir,
        
        # Training parameters
        num_train_epochs=training_config['num_epochs'],
        per_device_train_batch_size=training_config['batch_size'],
        per_device_eval_batch_size=training_config['batch_size'],
        gradient_accumulation_steps=training_config.get('gradient_accumulation_steps', 1),
        
        # Learning rate and optimization
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        warmup_steps=training_config['warmup_steps'],
        lr_scheduler_type=training_config.get('lr_scheduler_type', 'linear'),
        
        # Evaluation and saving
        evaluation_strategy=training_config.get('evaluation_strategy', 'steps'),
        eval_steps=training_config.get('eval_steps', 500),
        save_strategy=training_config['save_strategy'],
        save_steps=training_config.get('save_steps', 500),
        save_total_limit=training_config['save_total_limit'],
        load_best_model_at_end=training_config['load_best_model_at_end'],
        metric_for_best_model=training_config['metric_for_best_model'],
        
        # Logging
        logging_steps=training_config['logging_steps'],
        logging_first_step=training_config.get('logging_first_step', True),
        
        # Hardware optimization
        fp16=hardware_config.get('mixed_precision', False) == 'fp16',
        bf16=hardware_config.get('mixed_precision', False) == 'bf16',
        dataloader_num_workers=hardware_config['dataloader_num_workers'],
        dataloader_pin_memory=hardware_config['pin_memory'],
        
        # Additional settings
        remove_unused_columns=False,  # Keep for custom data handling
        push_to_hub=False,
        report_to=["wandb"] if config.get('experiment', {}).get('wandb', {}).get('enabled', False) else None,
        run_name=config.get('experiment', {}).get('name', 'trocr_cds_training'),
        
        # Regularization
        max_grad_norm=training_config.get('max_grad_norm', 1.0),
        dataloader_drop_last=training_config.get('dataloader_drop_last', False),
        
        # Early stopping
        greater_is_better=False  # For loss-based metrics
    )
    
    return args


def setup_callbacks(config: Dict) -> list:
    """Setup training callbacks."""
    callbacks = []
    
    # Early stopping
    early_stopping_config = config['training']
    if early_stopping_config.get('early_stopping_patience'):
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=early_stopping_config['early_stopping_patience'],
                early_stopping_threshold=early_stopping_config.get('early_stopping_threshold', 0.0)
            )
        )
    
    return callbacks


if __name__ == "__main__":
    # Test model creation
    model = TrOCRForCdS()
    info = model.get_model_info()
    print("Model Info:")
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 384, 384)
    output = model.generate(dummy_input, max_length=50)
    print(f"Generated shape: {output.shape}")
