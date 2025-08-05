import os
import torch
import logging
import yaml
from transformers import CLIPTokenizer
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
from diffusers.quantizers import PipelineQuantizationConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import math

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    if torch.xpu.is_available():
        dev = torch.device("xpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {dev}")
    return dev

class ImageDataset(Dataset):
    def __init__(self, folder, tokenizer, trigger_token, size=512):
        self.files = [f for f in os.listdir(folder)
                      if f.lower().endswith(('.png','jpg','jpeg'))]
        self.tokenizer = tokenizer
        self.trigger = trigger_token
        self.size = size
        self.transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5]*3, [0.5]*3)
        ])
        self.folder = folder
    def __len__(self):
        return len(self.files)
    def __getitem__(self, i):
        path = os.path.join(self.folder, self.files[i])
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        prompt = f"a photo of {self.trigger}"
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids[0]
        return img, input_ids

class CosineAnnealingWarmRestScheduler:
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.min_lr = float(config['min_lr'])
        self.max_lr = float(config['max_lr'])
        self.warmup_steps = int(config['warmup_steps'])
        self.restart_interval = int(config.get('restart_interval', 1000))
        self.restart_mult = float(config.get('restart_mult', 1.0))
        self.step_count = 0
        self.cycle_step = 0
        self.current_cycle_length = self.restart_interval
        self.cycle_count = 0

    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            progress = min(1.0, self.step_count / max(1, self.warmup_steps))
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            if self.cycle_step >= self.current_cycle_length:
                self.cycle_step = 0
                self.cycle_count += 1
                self.current_cycle_length = max(1, int(self.current_cycle_length * self.restart_mult))
            
            progress = self.cycle_step / max(1, self.current_cycle_length)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            
            self.cycle_step += 1
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def train(config_path):
    config = load_config(config_path)
    training_config = config['training']
    scheduler_config = config['scheduler']
    model_config = config['model']
    lora_config = config['lora']
    early_stopping_config = config['early_stopping']

    device = get_device()
    logger.info("Loading tokenizer...")
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    logger.info(f"Loading Stable Diffusion {model_config['pretrained_model_name']}")
    quantization_config = None
    if model_config.get('load_in_low_bit', False):
        quantization_config = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["text_encoder", "vae"]
        )

    pipe = StableDiffusionPipeline.from_pretrained(
        model_config['pretrained_model_name'],
        variant=model_config['variant'],
        torch_dtype=torch.float16,
        use_safetensors=model_config['use_safetensors'],
        safety_checker=model_config['safety_checker'],
        quantization_config=quantization_config
    ).to(device)

    scheduler = EulerDiscreteScheduler.from_pretrained(
        model_config['pretrained_model_name'], subfolder="scheduler"
    )

    text_encoder = pipe.text_encoder.eval()
    vae = pipe.vae
    unet = pipe.unet

    logger.info("Setting up LoRA on UNet...")
    peft_config = LoraConfig(
        task_type=getattr(TaskType, lora_config['task_type']),
        inference_mode=lora_config['inference_mode'],
        r=lora_config['r'],
        lora_alpha=lora_config['lora_alpha'],
        target_modules=lora_config['target_modules'],
    )
    unet = get_peft_model(unet, peft_config).to(device)
    unet.train()

    ds = ImageDataset(training_config['folder'], tokenizer, training_config['trigger_token'], size=training_config['image_size'])
    loader = DataLoader(ds, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(unet.parameters(), lr=float(training_config['learning_rate']))
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    lr_scheduler = CosineAnnealingWarmRestScheduler(optimizer, scheduler_config)

    best_loss = float('inf')
    patience = early_stopping_config['patience']
    epochs_without_improvement = 0

    logger.info("Beginning training loop...")
    global_step = 0
    for epoch in range(training_config['epochs']):
        logger.info(f"Epoch {epoch + 1}/{training_config['epochs']}")
        epoch_loss = 0.0
        progress_bar = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
        for step, (img, input_ids) in enumerate(progress_bar):
            img = img.to(device)
            input_ids = input_ids.to(device)

            with torch.amp.autocast('xpu', dtype=torch.float16):
                latents = vae.encode(img * 2 - 1).latent_dist.sample() * pipe.vae.config.scaling_factor
                noise = torch.randn_like(latents)
                t = torch.randint(0, scheduler.config.num_train_timesteps, (latents.size(0),), device=device).long()
                noisy = scheduler.add_noise(latents, noise, t)

                enc = text_encoder(input_ids)[0]
                pred = unet(noisy, t, encoder_hidden_states=enc).sample
                loss = torch.nn.functional.mse_loss(pred, noise)

            if loss.dim() > 0:
                loss = loss.mean()

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            # Store the loss value before deleting the tensor
            loss_value = loss.item()
            epoch_loss += loss_value
            current_lr = lr_scheduler.step()
            
            progress_bar.set_postfix(loss=loss_value, lr=current_lr, step=global_step)

            # Explicitly delete intermediate tensors
            del latents, noise, t, noisy, enc, pred, loss

            global_step += 1
  
            # Clear cache more frequently
            if step % 10 == 0:
                torch.xpu.empty_cache()

        avg_epoch_loss = epoch_loss / len(loader)
        logger.info(f"Epoch {epoch + 1} — Avg Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                logger.info(f"Early stopping triggered at epoch {epoch + 1}")
                break

        torch.xpu.empty_cache()

    logger.info("Saving LoRA adapters...")
    output_folder = config.training.get("output_folder")
    unet.save_pretrained(output_folder)
    logger.info(f"Saved adapters to ./{output_folder}")

if __name__ == "__main__":
    logger.info("Script start")
    train(config_path=".\src\config\sdxl-turbo_lora_xpu.yaml")
    logger.info("Script done")