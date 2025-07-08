# from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch
from ipex_llm.transformers import AutoModelForCausalLM

try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False

from ipex_llm.transformers.qlora import prepare_model_for_kbit_training

def create_model(config):
    return create_model_config(config['model'])

def create_model_config(model_config):
    """
    Create and return a model based on the specified configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        model: An instance of the specified model architecture.
    """
    model_id = model_config['model_id']
    device = torch.device("xpu" if model_config.get('ipex', {}).get('enabled', False) and torch.xpu.is_available() else "cpu")

    # Get from_pretrained_params as a dict from config
    pretrained_kwargs = model_config.get("from_pretrained_params", {})

    # Convert torch_dtype string to actual torch dtype if present
    if "torch_dtype" in pretrained_kwargs and isinstance(pretrained_kwargs["torch_dtype"], str):
        pretrained_kwargs["torch_dtype"] = getattr(torch, pretrained_kwargs["torch_dtype"])

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        **pretrained_kwargs
    ).to(device)
    # Optionally prepare for kbit training
    if model_config.get('prepare_kbit_training', False):
        model = prepare_model_for_kbit_training(model)
    if model_config.get('gradient_checkpointing', False):
        model.gradient_checkpointing_enable()
        print("[XPU] Gradient checkpointing enabled")
    model.train()
    # Apply LoRA if enabled
    lora_cfg = model_config.get('lora', {})
    if lora_cfg.get('enabled', False):
        lora_config = LoraConfig(
            r=lora_cfg['r'],
            lora_alpha=lora_cfg['lora_alpha'],
            target_modules=lora_cfg['target_modules'],
            lora_dropout=lora_cfg['lora_dropout'],
            bias=lora_cfg['bias']
        )
        model = get_peft_model(model, lora_config)
    # Apply IPEX optimization if enabled
    if model_config.get('ipex', {}).get('enabled', False) and has_ipex:
        model = ipex.optimize(model.eval())
    # Gradient checkpointing
    return (model, device)