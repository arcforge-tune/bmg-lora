from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
import torch

try:
    import intel_extension_for_pytorch as ipex
    has_ipex = True
except ImportError:
    has_ipex = False

from ipex_llm.transformers.qlora import prepare_model_for_kbit_training

def create_model(model_config):
    """
    Create and return a model based on the specified configuration.

    Args:
        config (dict): Configuration dictionary containing model parameters.

    Returns:
        model: An instance of the specified model architecture.
    """
    model_id = model_config['model_id']
    device = torch.device("xpu" if model_config.get('ipex', {}).get('enabled', False) and torch.xpu.is_available() else "cpu")
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    # Apply IPEX optimization if enabled
    if model_config.get('ipex', {}).get('enabled', False) and has_ipex:
        model = ipex.optimize(model.eval())
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
    # Optionally prepare for kbit training
    if model_config.get('prepare_kbit_training', False):
        model = prepare_model_for_kbit_training(model)
    return model