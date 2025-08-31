@echo off
REM Example script to run TrOCR training pipeline on Windows

echo 🚀 TrOCR Training Pipeline for Constance de Salm
echo ==============================================

REM Configuration
set CONFIG_DIR=..\config
set DATA_DIR=..\data
set MODELS_DIR=..\models
set OUTPUTS_DIR=..\outputs

REM Create directories
if not exist %DATA_DIR% mkdir %DATA_DIR%
if not exist %MODELS_DIR% mkdir %MODELS_DIR%
if not exist %OUTPUTS_DIR% mkdir %OUTPUTS_DIR%

echo.
echo 📊 Step 1: Data Preparation
echo ---------------------------
python prepare_data.py ^
    --config %CONFIG_DIR%\base_config.yaml ^
    --output_dir %DATA_DIR% ^
    --log_level INFO

if %errorlevel% neq 0 (
    echo ❌ Data preparation failed
    exit /b 1
)

echo.
echo 🎯 Step 2: Training
echo ------------------
python train.py ^
    --config %CONFIG_DIR%\training_config.yaml ^
    --base_config %CONFIG_DIR%\base_config.yaml ^
    --output_dir %MODELS_DIR% ^
    --experiment_name trocr_cds_%date:~10,4%%date:~4,2%%date:~7,2%_%time:~0,2%%time:~3,2%%time:~6,2% ^
    --log_level INFO

if %errorlevel% neq 0 (
    echo ❌ Training failed
    exit /b 1
)

echo.
echo 📈 Step 3: Evaluation
echo --------------------
python evaluate.py ^
    --model_path %MODELS_DIR%\final ^
    --config %CONFIG_DIR%\base_config.yaml ^
    --output_dir %OUTPUTS_DIR%\evaluation ^
    --save_predictions ^
    --log_level INFO

if %errorlevel% neq 0 (
    echo ❌ Evaluation failed
    exit /b 1
)

echo.
echo 🧪 Step 4: Inference Demo
echo ------------------------
python inference.py ^
    --model_path %MODELS_DIR%\final ^
    --config %CONFIG_DIR%\base_config.yaml ^
    --input_dir %DATA_DIR%\test_samples ^
    --output_dir %OUTPUTS_DIR%\predictions ^
    --log_level INFO

if %errorlevel% neq 0 (
    echo ❌ Inference failed
    exit /b 1
)

echo.
echo ✅ Pipeline completed successfully!
echo Results can be found in: %OUTPUTS_DIR%
pause
