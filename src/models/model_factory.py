import torch
import os
from peft import get_peft_model, set_peft_model_state_dict, LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False

def create_model(config, checkpoint_path=None):
    return create_model_config(config['model'], checkpoint_path)

def create_model_config(model_config, checkpoint_path):
    model_id = model_config['model_id']
    device = torch.device("xpu" if model_config.get('ipex', {}).get('enabled', False) and torch.xpu.is_available() else "cpu")
    pretrained_kwargs = model_config.get("from_pretrained_params", {})

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=False,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        llm_int8_skip_modules=pretrained_kwargs.get('modules_to_not_convert', [])
    )

    if model_config.get('load_in_4bit', True):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            low_cpu_mem_usage=True,
            quantization_config=bnb_config,
            torch_dtype=torch.float16,
            device_map={'': torch.xpu.current_device()} if device.type == "xpu" else None,
            use_cache=pretrained_kwargs.get('use_cache', False),
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map={'': torch.xpu.current_device()} if device.type == "xpu" else None,
            use_cache=pretrained_kwargs.get('use_cache', False),
        )

    model = prepare_model_for_kbit_training(model)

    if model_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        print("[XPU] Gradient checkpointing enabled")

    lora_cfg = model_config.get('lora', {})
    if lora_cfg.get('enabled', False):
        lora_config = LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['lora_alpha'],
            target_modules=lora_cfg['target_modules'],
            lora_dropout=lora_cfg['lora_dropout'],
            bias=lora_cfg['bias'],
            task_type="CAUSAL_LM"
        )
        model = get_peft_model(model, lora_config)

        if checkpoint_path:
            adapter_path = os.path.join(checkpoint_path, "adapter_model.bin")
            if os.path.exists(adapter_path):
                print(f"Loading adapter weights from {adapter_path}")
                adapter_weights = torch.load(adapter_path)
                set_peft_model_state_dict(model, adapter_weights)

    if model_config.get('ipex', {}).get('optimize_model', False) and has_ipex:
        model = ipex.optimize(model.eval(), dtype=torch.int4)
    return (model, device)