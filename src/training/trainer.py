import torch
from torch.optim import AdamW
from tqdm import tqdm


class ModelInstructions:
    def batch_to_device(self, batch, device):
        # Default: move all tensors in dict to device
        return {k: v.to(device) for k, v in batch.items()}


class CustomInstructions(ModelInstructions):
    pass  # Can override batch_to_device if needed


class Trainer:
    def __init__(self, model, train_loader, val_loader, config, tokenizer=None, instructions=None):
        # Accepts either a full config dict or split configs
        if 'training' in config and 'model' in config and 'lora' in config['model']:
            configTrain = config['training']
            configLora = config['model']['lora']
        else:
            configTrain = config
            configLora = {}
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.configTrain = configTrain
        self.configLora = configLora
        self.tokenizer = tokenizer
        self.device = torch.device(
            "xpu" if configTrain.get('device', 'auto') == 'xpu' and hasattr(torch, 'xpu') and torch.xpu.is_available()
            else "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model.to(self.device)
        optimizer_kwargs = {
            "params": self.model.parameters(),
            "lr": float(configTrain['learning_rate'])
        }
        if 'weight_decay' in configTrain:
            optimizer_kwargs["weight_decay"] = float(configTrain['weight_decay'])
        self.optimizer = AdamW(**optimizer_kwargs)
        self.instructions = instructions or ModelInstructions()

    def train(self, use_amp=False, grad_accum_steps=1, save_steps=None, save_checkpoint_fn=None, use_tqdm=False):
        epochs = self.configTrain['epochs']
        global_step = 0
        total_steps = epochs * (len(self.train_loader) // grad_accum_steps)
        for epoch in range(epochs):
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()
            loader = self.train_loader
            progress = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}") if use_tqdm else loader
            for step, batch in enumerate(progress):
                batch = self.instructions.batch_to_device(batch, self.device)
                if use_amp:
                    with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        scaled_loss = loss / grad_accum_steps
                    scaled_loss.backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    (loss / grad_accum_steps).backward()
                total_loss += loss.item()
                if (step + 1) % grad_accum_steps == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    if save_steps and save_checkpoint_fn and global_step % save_steps == 0:
                        save_checkpoint_fn(self.model, self.tokenizer, self.configTrain.get('output_dir', 'outputs/lora_finetuned'), global_step, epoch+1)
                if use_tqdm:
                    avg_loss = total_loss / (step + 1)
                    progress.set_postfix({'loss': f'{avg_loss:.4f}', 'step': f'{global_step}/{total_steps}'})
            epoch_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} completed - Avg Loss: {epoch_loss:.4f}")
            if(self.val_loader):
                self.validate(epoch)
            # Clear XPU cache after each epoch
            torch.xpu.empty_cache()
        if self.configTrain.get('save_model', True):
            self.save_model()

    def validate(self, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} — Val Loss: {avg_loss:.4f}")

    def save_model(self):
        output_dir = self.configTrain.get('output_dir', 'outputs/lora_finetuned')
        self.model.save_pretrained(output_dir)
        if self.tokenizer:
            self.tokenizer.save_pretrained(output_dir)
        print(f"\n💾 Model saved to {output_dir}")