#!/bin/bash
# Example script to run TrOCR training pipeline

set -e  # Exit on any error

echo "🚀 TrOCR Training Pipeline for Constance de Salm"
echo "=============================================="

# Configuration
CONFIG_DIR="../config"
DATA_DIR="../data"
MODELS_DIR="../models"
OUTPUTS_DIR="../outputs"

# Create directories
mkdir -p $DATA_DIR $MODELS_DIR $OUTPUTS_DIR

echo ""
echo "📊 Step 1: Data Preparation"
echo "---------------------------"
python prepare_data.py \
    --config $CONFIG_DIR/base_config.yaml \
    --output_dir $DATA_DIR \
    --log_level INFO

echo ""
echo "🎯 Step 2: Training"
echo "------------------"
python train.py \
    --config $CONFIG_DIR/training_config.yaml \
    --base_config $CONFIG_DIR/base_config.yaml \
    --output_dir $MODELS_DIR \
    --experiment_name "trocr_cds_$(date +%Y%m%d_%H%M%S)" \
    --log_level INFO

echo ""
echo "📈 Step 3: Evaluation"
echo "--------------------"
python evaluate.py \
    --model_path $MODELS_DIR/final \
    --config $CONFIG_DIR/base_config.yaml \
    --output_dir $OUTPUTS_DIR/evaluation \
    --save_predictions \
    --log_level INFO

echo ""
echo "🧪 Step 4: Inference Demo"
echo "------------------------"
python inference.py \
    --model_path $MODELS_DIR/final \
    --config $CONFIG_DIR/base_config.yaml \
    --input_dir $DATA_DIR/test_samples \
    --output_dir $OUTPUTS_DIR/predictions \
    --log_level INFO

echo ""
echo "✅ Pipeline completed successfully!"
echo "Results can be found in: $OUTPUTS_DIR"
