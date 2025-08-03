import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model
import logging
import yaml

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_config(config_path="src/config/stable-diffusion-2.1_lora_xpu.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def get_device():
    if torch.xpu.is_available():
        dev = torch.device("xpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {dev}")
    return dev

def load_lora_model(device, use_lora=True):
    config = load_config()
    model_config = config["model"]
    
    logger.info("Loading Stable Diffusion 2.1 base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_config["pretrained_model_name"],
        variant=model_config["variant"],
        torch_dtype=torch.float16,
        use_safetensors=model_config["use_safetensors"],
        safety_checker=model_config["safety_checker"]
    ).to(device)

    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)

    if use_lora:
        logger.info("Loading LoRA adapters...")
        peft_config = LoraConfig(
            task_type=config["lora"]["task_type"],
            inference_mode=True,
            r=config["lora"]["r"],
            lora_alpha=config["lora"]["lora_alpha"],
            target_modules=config["lora"]["target_modules"],
        )
        pipe.unet = get_peft_model(pipe.unet, peft_config)
        pipe.unet.load_adapter("lora_sd_small_xpu_2.1", "default")
        pipe.unet.eval()

    return pipe

def generate_images(pipe, prompt, num_images=1, size=768):
    logger.info(f"Generating images for prompt: '{prompt}'")
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=num_images,
        height=size,
        width=size,
        num_inference_steps=200,
        guidance_scale=7.0
        # ,generator=torch.Generator(pipe.device).manual_seed(42)
    ).images
    return images

def save_images(images, output_dir="./outputs/images"):
    import os
    os.makedirs(output_dir, exist_ok=True)
    for i, img in enumerate(images):
        img.save(f"{output_dir}/generated_{i}.png")
    logger.info(f"Saved images to {output_dir}")

if __name__ == "__main__":
    device = get_device()
    pipe = load_lora_model(device, use_lora=True)

    prompt = "arafed woman with long black hair in a car, 30 years old woman, female with long black hair, selfie of a young woman, 18 years old, alanis guillen, ayahausca, young woman with long dark hair, long hair and red shirt"
    images = generate_images(pipe, prompt, num_images=3, size=512)
    save_images(images)