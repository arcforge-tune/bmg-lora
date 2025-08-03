import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
from peft import LoraConfig, get_peft_model
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.xpu.is_available():
        dev = torch.device("xpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {dev}")
    return dev

def load_lora_model(device):
    logger.info("Loading Stable Diffusion v1.5 base model...")
    pipe = StableDiffusionPipeline.from_pretrained(
        "stable-diffusion-v1-5/stable-diffusion-v1-5",
        variant="fp16",
        torch_dtype=torch.float16,
        use_safetensors=True,
        safety_checker=None
    ).to(device)

    logger.info("Loading LoRA adapters...")
    peft_config = LoraConfig(
        task_type="TEXT_TO_IMAGE",  # Placeholder, adjust if needed
        inference_mode=True,
        r=4,
        lora_alpha=8,
        target_modules=["proj_in", "proj_out"],
    )
    pipe.unet = get_peft_model(pipe.unet, peft_config)
    pipe.unet.load_adapter("lora_sd_small_xpu", "default")  # Load saved adapters
    pipe.unet.eval()

    return pipe

def generate_images(pipe, prompt, num_images=1, size=512):
    logger.info(f"Generating images for prompt: '{prompt}'")
    images = pipe(
        prompt=prompt,
        num_images_per_prompt=num_images,
        height=size,
        width=size,
        num_inference_steps=300,
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
    pipe = load_lora_model(device)

    # Example prompt (replace with your desired prompt)
    prompt = "a stunning portrait of scarjo_style, elegant expression, wavy golden-blonde hair cascading over one shoulder, wearing a classic red evening gown with a deep neckline, diamond earrings catching soft light, posing against a softly blurred luxury interior background with warm ambient lighting. (Taken on: full-frame DSLR, studio lighting, shallow depth of field, ultra sharp focus, 8k resolution, professional fashion photography style)"
    images = generate_images(pipe, prompt, num_images=3, size=512)
    save_images(images)