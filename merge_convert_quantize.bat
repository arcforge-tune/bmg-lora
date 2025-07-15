@echo off
REM === Usage: merge_convert_quantize.bat --base_model "model_name" --lora_adapter_dir "path_to_lora_dir" ===

REM === Configuration Variables ===
set "PROJECT_ROOT=%cd%"
set "CONDA_ENV_MERGE=myenv"
set "CONDA_ENV_LLAMA=myenv"
set "LLAMA_ROOT=c:\llama.cpp"
set "LLAMA_BUILD=c:\llama.cpp\build\bin\Release"

REM === Parse Arguments ===
setlocal ENABLEDELAYEDEXPANSION
for %%A in (%*) do (
    if "%%~A"=="--base_model" (
        set "NEXT_IS_BASE=1"
        set "NEXT_IS_LORA="
    ) else if "%%~A"=="--lora_adapter_dir" (
        set "NEXT_IS_LORA=1"
        set "NEXT_IS_BASE="
    ) else if defined NEXT_IS_BASE (
        set "BASE_MODEL=%%~A"
        set "NEXT_IS_BASE="
    ) else if defined NEXT_IS_LORA (
        set "LORA_DIR=%%~A"
        set "NEXT_IS_LORA="
    )
)

REM === Validate input ===
if not defined BASE_MODEL (
    echo [ERROR] --base_model not provided.
    exit /b 1
)
if not defined LORA_DIR (
    echo [ERROR] --lora_adapter_dir not provided.
    exit /b 1
)

REM === Extract folder name from path ===
for %%F in ("%LORA_DIR%") do set "LORA_FOLDER=%%~nxF"
set "MERGED_DIR=outputs\%LORA_FOLDER%_merged"
set "GGUF_FILE=outputs\%LORA_FOLDER%_f16.gguf"
set "QUANT_FILE=outputs\%LORA_FOLDER%_q4_0.gguf"

echo ===============================
echo Step 1: Merging LoRA Adapter...
echo ===============================
call conda activate %CONDA_ENV_MERGE%
python src/merge_lora_and_export.py --base_model "%BASE_MODEL%" --lora_adapter_dir "%LORA_DIR%" --merged_output_dir "%MERGED_DIR%"
if errorlevel 1 (
    echo [ERROR] Merging failed.
    exit /b 1
)

echo ===============================
echo Step 2: Converting to GGUF...
echo ===============================
call conda activate %CONDA_ENV_LLAMA%
cd /d "%LLAMA_ROOT%"
python convert_hf_to_gguf.py "%PROJECT_ROOT%\%MERGED_DIR%" --outfile "%PROJECT_ROOT%\%GGUF_FILE%" --outtype f16
if errorlevel 1 (
    echo [ERROR] Conversion to GGUF failed.
    exit /b 1
)

echo ===============================
echo Step 3: Quantizing Model...
echo ===============================
cd /d "%LLAMA_BUILD%"
llama-quantize.exe "%PROJECT_ROOT%\%GGUF_FILE%" "%PROJECT_ROOT%\%QUANT_FILE%" Q4_0
if errorlevel 1 (
    echo [ERROR] Quantization failed.
    exit /b 1
)

echo ===============================
echo All steps completed successfully!
echo Merged Model: %MERGED_DIR%
echo GGUF File: %GGUF_FILE%
echo Quantized File: %QUANT_FILE%
echo ===============================
exit /b 0
