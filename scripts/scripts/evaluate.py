#!/usr/bin/env python3
"""
Evaluation script for trained TrOCR models.
"""

import argparse
import logging
import sys
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model import TrOCRForCdS
from src.evaluator import HTREvaluator
from src.data_loader import CdSDataLoader
from src.utils import load_config, setup_logging, save_json, get_device

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained TrOCR model on test data"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model directory"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--test_dir", 
        type=str,
        help="Directory containing test data (overrides config)"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/evaluation/",
        help="Output directory for evaluation results"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--save_predictions", 
        action="store_true",
        help="Save all predictions to file"
    )
    parser.add_argument(
        "--compare_with_kraken", 
        action="store_true",
        help="Compare results with existing Kraken predictions"
    )
    parser.add_argument(
        "--kraken_predictions_dir", 
        type=str,
        help="Directory containing Kraken predictions for comparison"
    )
    parser.add_argument(
        "--log_level", 
        type=str, 
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    # Load configuration
    try:
        config = load_config(args.config)
        logger.info(f"Loaded configuration from {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override batch size
    config['training']['batch_size'] = args.batch_size
    
    # Override test directory if provided
    if args.test_dir:
        config['data']['source_dirs'] = [args.test_dir]
        logger.info(f"Using test directory: {args.test_dir}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Get device
        device = get_device()
        logger.info(f"Using device: {device}")
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        model = TrOCRForCdS.load_model(str(model_path), config)
        model.to(device)
        model.eval()
        
        # Print model info
        model_info = model.get_model_info()
        logger.info("Model Information:")
        for key, value in model_info.items():
            logger.info(f"  {key}: {value}")
        
        # Initialize data loader
        logger.info("Loading test data...")
        data_loader = CdSDataLoader(config)
        _, _, test_dataset = data_loader.create_datasets()
        
        if len(test_dataset) == 0:
            logger.error("No test data found")
            return 1
        
        logger.info(f"Test dataset size: {len(test_dataset)}")
        
        # Initialize evaluator
        evaluator = HTREvaluator(config.get('evaluation', {}))
        
        # Run inference on test set
        logger.info("Running inference on test set...")
        predictions = []
        references = []
        image_paths = []
        
        import torch
        from torch.utils.data import DataLoader
        from src.data_loader import collate_fn
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            collate_fn=collate_fn
        )
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                logger.info(f"Processing batch {batch_idx + 1}/{len(test_loader)}")
                
                pixel_values = batch['pixel_values'].to(device)
                batch_texts = batch['texts']
                batch_paths = batch['image_paths']
                
                # Generate predictions
                generated_ids = model.generate(
                    pixel_values, 
                    max_length=384,
                    num_beams=5,
                    early_stopping=True
                )
                
                # Decode predictions
                batch_predictions = model.processor.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True
                )
                
                predictions.extend(batch_predictions)
                references.extend(batch_texts)
                image_paths.extend(batch_paths)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        # Compute metrics
        logger.info("Computing evaluation metrics...")
        metrics = evaluator.compute_metrics(predictions, references)
        
        # Generate detailed report
        detailed_report = evaluator.generate_detailed_report(
            predictions, references, image_paths
        )
        
        # Print results
        logger.info("Evaluation Results:")
        logger.info("=" * 50)
        for metric_name, value in metrics.items():
            logger.info(f"{metric_name:20s}: {value:.4f}")
        
        # Save results
        results = {
            'model_path': str(model_path),
            'test_dataset_size': len(test_dataset),
            'metrics': metrics,
            'detailed_report': detailed_report,
            'config': config
        }
        
        results_file = output_dir / "evaluation_results.json"
        save_json(results, str(results_file))
        logger.info(f"Results saved to {results_file}")
        
        # Save predictions if requested
        if args.save_predictions:
            predictions_data = []
            for i, (pred, ref, img_path) in enumerate(zip(predictions, references, image_paths)):
                predictions_data.append({
                    'index': i,
                    'image_path': img_path,
                    'prediction': pred,
                    'reference': ref,
                    'cer': evaluator.compute_cer(pred, ref),
                    'wer': evaluator.compute_wer(pred, ref)
                })
            
            predictions_file = output_dir / "predictions.json"
            save_json(predictions_data, str(predictions_file))
            logger.info(f"Predictions saved to {predictions_file}")
        
        # Compare with Kraken if requested
        if args.compare_with_kraken and args.kraken_predictions_dir:
            logger.info("Comparing with Kraken predictions...")
            kraken_comparison = compare_with_kraken_predictions(
                predictions, references, image_paths,
                args.kraken_predictions_dir, evaluator
            )
            
            comparison_file = output_dir / "kraken_comparison.json"
            save_json(kraken_comparison, str(comparison_file))
            logger.info(f"Kraken comparison saved to {comparison_file}")
            
            # Print comparison summary
            logger.info("TrOCR vs Kraken Comparison:")
            logger.info(f"TrOCR CER: {kraken_comparison['trocr_metrics']['cer']:.4f}")
            logger.info(f"Kraken CER: {kraken_comparison['kraken_metrics']['cer']:.4f}")
            improvement = kraken_comparison['kraken_metrics']['cer'] - kraken_comparison['trocr_metrics']['cer']
            logger.info(f"CER Improvement: {improvement:.4f} ({'better' if improvement > 0 else 'worse'})")
        
        logger.info("Evaluation completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


def compare_with_kraken_predictions(trocr_predictions, references, image_paths, 
                                  kraken_dir, evaluator):
    """Compare TrOCR predictions with Kraken predictions."""
    logger.info("Loading Kraken predictions for comparison...")
    
    kraken_predictions = []
    kraken_dir = Path(kraken_dir)
    
    for img_path in image_paths:
        img_name = Path(img_path).stem
        
        # Look for corresponding Kraken prediction file
        kraken_file_candidates = [
            kraken_dir / f"{img_name}.txt",
            kraken_dir / f"{img_name}.xml",
            kraken_dir / f"{img_name}_prediction.txt"
        ]
        
        kraken_text = ""
        for candidate in kraken_file_candidates:
            if candidate.exists():
                if candidate.suffix == '.txt':
                    with open(candidate, 'r', encoding='utf-8') as f:
                        kraken_text = f.read().strip()
                elif candidate.suffix == '.xml':
                    # Extract text from XML (simplified)
                    with open(candidate, 'r', encoding='utf-8') as f:
                        content = f.read()
                        import re
                        text_matches = re.findall(r'CONTENT="([^"]*)"', content)
                        kraken_text = ' '.join(text_matches)
                break
        
        kraken_predictions.append(kraken_text)
    
    # Compute metrics for both
    trocr_metrics = evaluator.compute_metrics(trocr_predictions, references)
    kraken_metrics = evaluator.compute_metrics(kraken_predictions, references)
    
    # Sample comparisons
    sample_comparisons = []
    for i in range(min(10, len(trocr_predictions))):
        sample_comparisons.append({
            'index': i,
            'image_path': image_paths[i],
            'reference': references[i],
            'trocr_prediction': trocr_predictions[i],
            'kraken_prediction': kraken_predictions[i],
            'trocr_cer': evaluator.compute_cer(trocr_predictions[i], references[i]),
            'kraken_cer': evaluator.compute_cer(kraken_predictions[i], references[i])
        })
    
    return {
        'trocr_metrics': trocr_metrics,
        'kraken_metrics': kraken_metrics,
        'sample_comparisons': sample_comparisons,
        'total_samples': len(trocr_predictions)
    }


if __name__ == "__main__":
    exit(main())
