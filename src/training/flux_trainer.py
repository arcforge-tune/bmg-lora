import os
import torch
from transformers import CLIPTokenizer
from diffusers import FluxPipeline, DDIMScheduler
from diffusers.quantizers import PipelineQuantizationConfig
from diffusers import BitsAndBytesConfig as DiffusersBnbConfig
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_device():
    logger.info("Checking available devices...")
    device = torch.device("xpu" if torch.xpu.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    return device

class ImageDataset(Dataset):
    def __init__(self, folder, tokenizer, trigger_token, size=512):
        self.folder = folder
        self.tokenizer = tokenizer
        self.trigger_token = trigger_token
        self.size = size
        self.image_files = [f for f in os.listdir(folder) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = T.Compose([
            T.Resize(size),
            T.CenterCrop(size),
            T.ToTensor(),
            T.Normalize([0.5], [0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        prompt = f"a photo of {self.trigger_token}"
        input_ids = self.tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=self.tokenizer.model_max_length,
            return_tensors="pt"
        ).input_ids
        return image, input_ids[0]
def train(folder, trigger_token="xtoken", steps=1000, lr=4e-4, size=512):
    try:
        logger.info("Starting training process...")
        device = get_device()

        logger.info("Loading tokenizer...")
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

        logger.info("Creating pipeline quantization config...")
        pipeline_quant = PipelineQuantizationConfig(
            quant_backend="bitsandbytes_4bit",
            quant_kwargs={
                "load_in_4bit": True,
                "bnb_4bit_quant_type": "nf4",
                "bnb_4bit_compute_dtype": torch.bfloat16,
            },
            components_to_quantize=["transformer", "text_encoder_2"]
        )

        logger.info("Loading FluxPipeline with quantization...")
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-dev",
            quantization_config=pipeline_quant,
            torch_dtype=torch.bfloat16,
            use_auth_token=True
        ).to(device)

        logger.info("Extracting pipeline components...")
        text_encoder = pipe.text_encoder.eval()
        vae = pipe.vae
        unet = pipe.unet

        logger.info("Attaching LoRA to UNet...")
        peft_config = LoraConfig(
            task_type=TaskType.IMAGE_TEXT_TO_IMAGE,
            inference_mode=False,
            r=8, lora_alpha=8,
            target_modules=["proj_in", "proj_out"],
        )
        unet = get_peft_model(unet, peft_config)
        unet.train()

        logger.info("Preparing dataset and data loader...")
        ds = ImageDataset(folder, tokenizer, trigger_token, size=size)
        loader = DataLoader(ds, batch_size=1, shuffle=True)
        optimizer = torch.optim.Adam(unet.parameters(), lr=lr)
        scaler = torch.amp.GradScaler(enabled=(device.type in ["xpu"]))
        scheduler = DDIMScheduler.from_pretrained("black-forest-labs/FLUX.1-dev", subfolder="scheduler")

        logger.info("Starting training loop...")
        for step, (img, input_ids) in enumerate(loader):
            if step >= steps:
                break
            logger.info(f"Processing step {step}...")
            img = img.to(device); input_ids = input_ids.to(device)

            with torch.amp.autocast(device_type=device.type, dtype=torch.bfloat16):
                logger.info("Encoding image to latents...")
                latents = vae.encode(img * 2 - 1).latent_dist.sample().mul(0.18215)
                noise = torch.randn_like(latents)
                t = torch.randint(0, scheduler.num_train_timesteps,
                                (latents.size(0),), device=device).long()
                noisy = scheduler.add_noise(latents, noise, t)

                logger.info("Encoding text inputs...")
                enc = text_encoder(input_ids)[0]
                logger.info("Predicting noise...")
                pred = unet(noisy, t, enc).sample
                loss = torch.nn.functional.mse_loss(pred, noise)

            logger.info("Updating model weights...")
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            if step % 100 == 0:
                logger.info(f"Step {step} — loss {loss.item():.6f}")

        logger.info("Saving LoRA adapters...")
        unet.save_pretrained("lora_flux_nf4")
        logger.info("Saved LoRA adapters to ./lora_flux_nf4")

    except Exception as e:
        logger.error(f"An error occurred during training: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    logger.info("Script started.")
    train("path_to_your_folder")
    logger.info("Script completed successfully.")
