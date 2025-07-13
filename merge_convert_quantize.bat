@echo off
setlocal enabledelayedexpansion

:: Configuration
set BASE_DIR=%~dp0
:: install steps for llama.cpp
:: git clone https://github.com/ggerganov/llama.cpp
:: cd llama.cpp
:: cmake . -B build
:: cmake --build build --config Release
set LLAMA_PATH=c:\llm\llama.cpp
set CONDA_MERGE_ENV=test-qlora-ipex
set CONDA_LLAMA_ENV=llama-convert
set BUILD_DIR=%LLAMA_PATH%\build\bin\Release

:: Parse arguments
set "BASE_MODEL="
set "LORA_DIR="

:parse_args
if "%~1"=="" goto end_args
if "%~1"=="--base_model" (
    set "BASE_MODEL=%~2"
    shift
) else if "%~1"=="--lora_adapter_dir" (
    set "LORA_DIR=%~2"
    shift
)
shift
goto parse_args
:end_args

:: Validate arguments
if "!BASE_MODEL!"=="" (
    echo Error: Missing --base_model argument
    exit /b 1
)
if "!LORA_DIR!"=="" (
    echo Error: Missing --lora_adapter_dir argument
    exit /b 1
)

:: Generate output names
for %%A in ("%LORA_DIR%") do set "FOLDER_NAME=%%~nxA"
set MERGED_OUTPUT=%BASE_DIR%outputs\%FOLDER_NAME%_merged
set GGUF_F16=%BASE_DIR%outputs\%FOLDER_NAME%_f16.gguf
set GGUF_Q4=%BASE_DIR%outputs\%FOLDER_NAME%_q4_0.gguf

:: Step 1: Merge LoRA with base model
echo [1/3] Merging LoRA adapter...
call conda activate %CONDA_MERGE_ENV%
python "%BASE_DIR%src\merge_lora_and_export.py" ^
    --base_model "%BASE_MODEL%" ^
    --lora_adapter_dir "%LORA_DIR%" ^
    --merged_output_dir "%MERGED_OUTPUT%"
if %errorlevel% neq 0 exit /b %errorlevel%

:: Step 2: Convert to GGUF F16
echo [2/3] Converting to GGUF F16...
call conda activate %CONDA_LLAMA_ENV%
python "%LLAMA_PATH%\convert_hf_to_gguf.py" ^
    "%MERGED_OUTPUT%" ^
    --outfile "%GGUF_F16%" ^
    --outtype f16
if %errorlevel% neq 0 exit /b %errorlevel%

:: Step 3: Quantize to Q4_0
echo [3/3] Quantizing to Q4_0...
"%BUILD_DIR%\llama-quantize.exe" ^
    "%GGUF_F16%" ^
    "%GGUF_Q4%" ^
    Q4_0
if %errorlevel% neq 0 exit /b %errorlevel%

echo All operations completed successfully!
echo Merged model: %MERGED_OUTPUT%
echo F16 GGUF: %GGUF_F16%
echo Q4_0 GGUF: %GGUF_Q4%