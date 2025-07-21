import argparse
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import intel_extension_for_pytorch as ipex

# ----------- CONFIG ----------- #
BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
LORA_ADAPTER_DIR = "outputs/lora_llama3_1_8b_instruct_xpu"
MERGED_MODEL_OUTPUT = "outputs/lora_llama3_1_8b_instruct_xpu_merged"

# ----------- SETUP ARGUMENTS ----------- #
parser = argparse.ArgumentParser(description='Merge LoRA and Export')
parser.add_argument('--base_model', type=str, required=True, help='Base model name or path')
parser.add_argument('--lora_adapter_dir', type=str, required=True, help='LoRA adapter directory')
parser.add_argument('--merged_output_dir', type=str, required=True, help='Merged output directory')
parser.add_argument('--device_type', choices=['cpu', 'xpu'], default='cpu',
                   help='Device type to use (cpu or xpu)')
parser.add_argument('--optimize', type=bool, default=False,
                   help='Whether to apply IPEX optimization')

args = parser.parse_args()

# Set device based on argument
device = torch.device('xpu' if args.device_type == 'xpu' and torch.xpu.is_available() else 'cpu')

def merge_lora_and_export():
    print(f"[{device}] Loading full-precision base model for merging...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float32,           # Full precision required for merge
        trust_remote_code=True,
        use_cache=True,
        device_map={'': torch.xpu.current_device()} if device.type == "xpu" else None
    )
    
    # model = model.to(device)

    print(f"[{device}] Loading and attaching LoRA adapter...")
    model = PeftModel.from_pretrained(model, args.lora_adapter_dir).to(device)

    print(f"[{device}] Merging LoRA weights into base model...") 
    model = model.merge_and_unload()

    if args.optimize and device.type == "xpu":
        print("[XPU] Applying IPEX optimization for inference...")
        optimized_model = ipex.optimize(model.eval(), dtype=torch.bfloat16)
        model = optimized_model

    print(f"[{device}] Saving merged model to {args.merged_output_dir}...")
    os.makedirs(args.merged_output_dir, exist_ok=True)
    model.save_pretrained(args.merged_output_dir)

    print("[XPU] Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.lora_adapter_dir)
    tokenizer.save_pretrained(args.merged_output_dir)

    print(f"[{device}] ✅ Merge complete! Model and tokenizer saved.")

if __name__ == '__main__':
    merge_lora_and_export()