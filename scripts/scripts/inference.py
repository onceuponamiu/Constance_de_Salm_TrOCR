#!/usr/bin/env python3
"""
Inference script for TrOCR model on individual images.
"""

import argparse
import logging
import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.model import TrOCRForCdS
from src.utils import load_config, setup_logging, get_device, save_json
from PIL import Image
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_and_preprocess_image(image_path: str, target_size: tuple = (384, 384)) -> Image.Image:
    """
    Load and preprocess image for TrOCR inference.
    
    Args:
        image_path: Path to image file
        target_size: Target size for resizing
        
    Returns:
        Preprocessed PIL Image
    """
    image = Image.open(image_path).convert('RGB')
    
    # Resize while maintaining aspect ratio
    img_ratio = image.width / image.height
    target_ratio = target_size[0] / target_size[1]
    
    if img_ratio > target_ratio:
        # Image is wider than target
        new_width = target_size[0]
        new_height = int(new_width / img_ratio)
    else:
        # Image is taller than target
        new_height = target_size[1]
        new_width = int(new_height * img_ratio)
    
    # Resize and pad
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Create new image with target size and paste resized image
    new_image = Image.new('RGB', target_size, (255, 255, 255))
    paste_x = (target_size[0] - new_width) // 2
    paste_y = (target_size[1] - new_height) // 2
    new_image.paste(resized, (paste_x, paste_y))
    
    return new_image


def process_single_image(model: TrOCRForCdS, image_path: str, 
                        max_length: int = 384, num_beams: int = 5) -> dict:
    """
    Process a single image and return prediction with metadata.
    
    Args:
        model: Loaded TrOCR model
        image_path: Path to image file
        max_length: Maximum generation length
        num_beams: Number of beams for beam search
        
    Returns:
        Dictionary with prediction and metadata
    """
    start_time = time.time()
    
    # Load and preprocess image
    image = load_and_preprocess_image(image_path)
    
    # Run inference
    with torch.no_grad():
        pixel_values = model.processor(image, return_tensors="pt").pixel_values
        if torch.cuda.is_available():
            pixel_values = pixel_values.cuda()
        
        # Generate with different strategies
        predictions = {}
        
        # Greedy decoding
        greedy_ids = model.generate(pixel_values, max_length=max_length, do_sample=False)
        predictions['greedy'] = model.processor.batch_decode(greedy_ids, skip_special_tokens=True)[0]
        
        # Beam search
        beam_ids = model.generate(
            pixel_values, 
            max_length=max_length, 
            num_beams=num_beams,
            early_stopping=True
        )
        predictions['beam_search'] = model.processor.batch_decode(beam_ids, skip_special_tokens=True)[0]
        
        # Sampling (if different from greedy)
        if num_beams > 1:
            sample_ids = model.generate(
                pixel_values, 
                max_length=max_length, 
                do_sample=True,
                top_p=0.9,
                temperature=0.7
            )
            predictions['sampling'] = model.processor.batch_decode(sample_ids, skip_special_tokens=True)[0]
    
    inference_time = time.time() - start_time
    
    return {
        'image_path': image_path,
        'image_size': image.size,
        'predictions': predictions,
        'inference_time_seconds': inference_time,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }


def process_directory(model: TrOCRForCdS, input_dir: str, output_dir: str, 
                     image_extensions: list = None) -> list:
    """
    Process all images in a directory.
    
    Args:
        model: Loaded TrOCR model
        input_dir: Input directory containing images
        output_dir: Output directory for results
        image_extensions: List of valid image extensions
        
    Returns:
        List of results
    """
    if image_extensions is None:
        image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp']
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all image files
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_path.glob(f"*{ext}"))
        image_files.extend(input_path.glob(f"*{ext.upper()}"))
    
    logger.info(f"Found {len(image_files)} images in {input_dir}")
    
    results = []
    for i, image_file in enumerate(image_files):
        logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
        
        try:
            result = process_single_image(model, str(image_file))
            results.append(result)
            
            # Save individual result
            result_file = output_path / f"{image_file.stem}_result.json"
            save_json(result, str(result_file))
            
        except Exception as e:
            logger.error(f"Failed to process {image_file}: {e}")
            results.append({
                'image_path': str(image_file),
                'error': str(e),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run TrOCR inference on images"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained TrOCR model"
    )
    parser.add_argument(
        "--image_path", 
        type=str,
        help="Path to single image file"
    )
    parser.add_argument(
        "--input_dir", 
        type=str,
        help="Directory containing images to process"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="outputs/predictions/",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config", 
        type=str, 
        default="config/base_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--max_length", 
        type=int, 
        default=384,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--num_beams", 
        type=int, 
        default=5,
        help="Number of beams for beam search"
    )
    parser.add_argument(
        "--save_images", 
        action="store_true",
        help="Save processed images alongside results"
    )
    parser.add_argument(
        "--output_format", 
        type=str, 
        choices=["json", "txt", "csv"],
        default="json",
        help="Output format for results"
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
    
    # Validate arguments
    if not args.image_path and not args.input_dir:
        logger.error("Either --image_path or --input_dir must be specified")
        return 1
    
    if args.image_path and args.input_dir:
        logger.error("Cannot specify both --image_path and --input_dir")
        return 1
    
    # Check if model exists
    model_path = Path(args.model_path)
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        return 1
    
    try:
        # Load configuration
        config = load_config(args.config)
        
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
        logger.info(f"Model: {model_info['model_name']}")
        logger.info(f"Parameters: {model_info['trainable_parameters']:,}")
        
        # Create output directory
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        if args.image_path:
            # Process single image
            logger.info(f"Processing single image: {args.image_path}")
            
            if not Path(args.image_path).exists():
                logger.error(f"Image not found: {args.image_path}")
                return 1
            
            result = process_single_image(
                model, args.image_path, 
                max_length=args.max_length, 
                num_beams=args.num_beams
            )
            
            # Print results
            logger.info("Inference Results:")
            logger.info("=" * 50)
            for strategy, prediction in result['predictions'].items():
                logger.info(f"{strategy.upper():12s}: {prediction}")
            logger.info(f"Inference time: {result['inference_time_seconds']:.3f}s")
            
            # Save result
            if args.output_format == "json":
                output_file = output_dir / f"{Path(args.image_path).stem}_result.json"
                save_json(result, str(output_file))
            elif args.output_format == "txt":
                output_file = output_dir / f"{Path(args.image_path).stem}_result.txt"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result['predictions']['beam_search'])
            
            logger.info(f"Result saved to {output_file}")
            
        elif args.input_dir:
            # Process directory
            logger.info(f"Processing directory: {args.input_dir}")
            
            if not Path(args.input_dir).exists():
                logger.error(f"Directory not found: {args.input_dir}")
                return 1
            
            results = process_directory(model, args.input_dir, str(output_dir))
            
            # Save summary
            summary = {
                'total_images': len(results),
                'successful': len([r for r in results if 'predictions' in r]),
                'failed': len([r for r in results if 'error' in r]),
                'average_inference_time': sum(r.get('inference_time_seconds', 0) for r in results) / len(results) if results else 0,
                'results': results
            }
            
            summary_file = output_dir / "inference_summary.json"
            save_json(summary, str(summary_file))
            
            logger.info(f"Processed {summary['total_images']} images")
            logger.info(f"Successful: {summary['successful']}, Failed: {summary['failed']}")
            logger.info(f"Average inference time: {summary['average_inference_time']:.3f}s")
            logger.info(f"Summary saved to {summary_file}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        import traceback
        logger.debug(traceback.format_exc())
        return 1


if __name__ == "__main__":
    exit(main())
