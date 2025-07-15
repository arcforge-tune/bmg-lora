# ArcForge: Fine-Tuning on Intel Battlemage GPUs

ArcForge is an open-source toolkit for fine-tuning large language models (LLMs) using Intel Arc™ Battlemage and other next-generation Intel GPUs. It integrates cutting-edge tools like HuggingFace Transformers, Intel IPEX-LLM, and QLoRA/LoRA methods to deliver scalable, efficient training pipelines optimized for the Intel XPU architecture.

## 🚀 Project Goals

- Enable high-performance fine-tuning of open-weight LLMs on Intel GPUs
- Provide end-to-end workflows for QLoRA and LoRA with checkpointing, evaluation, and inference
- Support for modern instruction-tuning datasets (Alpaca, OpenOrca, etc.)
- Serve as a foundational framework for building ML infra on Intel's Battlemage GPUs

## 🧩 Key Features

- ✅ LoRA and QLoRA fine-tuning on LLaMA 2/3 models
- ✅ HuggingFace Transformers integration
- ✅ Intel IPEX-LLM acceleration for Arc GPUs
- ✅ Alpaca-compatible training datasets
- ✅ Efficient checkpointing, evaluation, and inference

## 📦 Installation

1. Clone the repository:
```bash
git clone https://github.com/arcforge-tune/bmg-lora.git
```

2. Install dependencies using conda and pip:
```bash
conda create -n fine-tune python=3.11.13
conda activate fine-tune
cd bmg-lora
pip install -r requirements.txt
```

3. Optional for fine-tune with meta-llama/Llama-3.1-8B-Instruct
install these packages:
```bash
tokenizers=0.21.2
transformers=4.53.1
```
## 🛠️ Usage

To run training with LLaMA-2 on the Alpaca dataset examples:

1. Clone the repository and install dependencies.

2. Configure your training parameters in `src/config/`

| File Name                     | Model ID        | Model Name              |
|------------------------------|-----------------|-----------------------|
| gpt2_lora_finetune_config.yaml  | gpt2           | GPT-2                 |
| llama2_hf_qlora_xpu_config.yaml | meta-llama/Llama-2-7b-hf   | Llama 2 (7B)          |
| llamma2_chat_hf_qlora_xpu_config.yaml | meta-llama/Llama-2-7b-chat-hf | Llama 2 (7B Chat)     |
| mistral-7B-v0.1_xpu_config.yaml  | mistralai/Mistral-7B-v0.1         | Mistral (7B)                |
| llama3.18B_qlora_config.yaml | meta-llama/Llama-3.1-8B-Instruct | llama 3.1 | 

3. Run the training for your model using the provided configuration:
### Start training:
```bash
python src/main.py --config .\src\config\gpt2_lora_finetune_config.yaml    
```
### Resume training from a checkpoint:
```bash
python src/main.py --config .\src\config\gpt2_lora_finetune_config.yaml --resume .\outputs\lora_llama3_1_8b_instruct_xpu\checkpoint-epoch1-step11\
```
### For a clean run or resume output use the PS script:
```bash
.\Run‑FineTune.ps1 -config .\src\config\gpt2_lora_finetune_config.yaml  
```
## Requirements

- Python 3.6+
- PyTorch
- HuggingFace Transformers
- Intel IPEX
- ONEDNN library

For GPU support, ensure you have the correct Intel GPU drivers installed
