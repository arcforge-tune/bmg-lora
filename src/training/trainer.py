import os
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
    def __init__(self, model, device, train_loader, val_loader, config, tokenizer=None, instructions=None):
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
        self.device = device
        optimizer_kwargs = {
            "params": self.model.parameters(),
            "lr": float(configTrain['learning_rate'])
        }
        if 'weight_decay' in configTrain:
            optimizer_kwargs["weight_decay"] = float(configTrain['weight_decay'])
        self.optimizer = AdamW(**optimizer_kwargs)
        self.instructions = instructions or ModelInstructions()
        # In Trainer constructor
        if 'scheduler' in configTrain:
            self.scheduler = CosineAnnealingWarmRestScheduler(self.optimizer, configTrain)
        else:
            self.scheduler = None  # Will use constant LR
        self.max_grad_norm = float(configTrain.get('max_grad_norm', 1.0))

    def train(self, use_amp=False, save_checkpoint_fn=None, use_tqdm=False):
        epochs = self.configTrain['epochs']
        grad_accum_steps = self.configTrain.get('gradient_accumulation_steps', 1)
        save_steps = self.configTrain.get('save_steps', 100)
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
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    self.optimizer.step()
                    
                     # Step scheduler after optimizer update
                    if self.scheduler is not None:
                        current_lr = self.scheduler.step()  # Get updated LR
                    else:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                    if save_steps and global_step % save_steps == 0:
                        if save_checkpoint_fn and callable(save_checkpoint_fn):
                            save_checkpoint_fn()
                        else:
                            self.save_model(global_step, epoch+1)
                if use_tqdm:
                    avg_loss = total_loss / (step + 1)
                    current_lr = self.optimizer.param_groups[0]['lr']
                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',  # Show current learning rate
                        'step': f'{global_step}/{total_steps}'
                    })
            epoch_loss = total_loss / len(loader)
            print(f"Epoch {epoch+1} completed - Avg Loss: {epoch_loss:.4f}")
            if(self.val_loader):
                self.validate(epoch, use_amp)
            # Clear XPU cache after each epoch
            torch.xpu.empty_cache()
        if self.configTrain.get('save_model', True):
            print("[XPU] Saving final LoRA adapter...")
            output_dir = self.configTrain.get('output_dir', 'outputs/lora_finetuned')
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"[XPU] Final model saved to {output_dir}")
            torch.xpu.empty_cache()

    def validate(self, epoch, use_amp):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device, non_blocking=True) for k, v in batch.items()}
                if use_amp:
                    with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                total_loss += loss.item()
        avg_loss = total_loss / len(self.val_loader)
        print(f"Epoch {epoch+1} — Val Loss: {avg_loss:.4f}")

    def save_model(self, global_step, epoch):
        output_dir = self.configTrain.get('output_dir', 'outputs/lora_finetuned')
        checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch{epoch}-step{global_step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(checkpoint_dir)
        print(f"Checkpoint saved at epoch {epoch}, step {global_step} -> {checkpoint_dir}")

import math

class CosineAnnealingWarmRestScheduler:
    def __init__(self, optimizer, configTrain):
        self.optimizer = optimizer
        scheduler_config = configTrain['scheduler']
        
        # Validate and set parameters
        self.min_lr = float(scheduler_config['min_lr'])
        self.max_lr = float(scheduler_config['max_lr'])
        self.warmup_steps = int(scheduler_config['warmup_steps'])
        self.restart_interval = int(scheduler_config.get('restart_interval', 1000))
        self.restart_mult = float(scheduler_config.get('restart_mult', 1.0))
        
        # Validate parameters
        if self.min_lr >= self.max_lr:
            raise ValueError("min_lr must be less than max_lr")
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if self.restart_interval <= 0:
            raise ValueError("restart_interval must be positive")
        if self.restart_mult <= 0:
            raise ValueError("restart_mult must be positive")
        
        # Initialize state
        self.step_count = 0
        self.cycle_step = 0
        self.current_cycle_length = self.restart_interval
        self.cycle_count = 0

    def step(self):
        self.step_count += 1
        
        # Warmup phase: Linear increase
        if self.step_count <= self.warmup_steps:
            progress = min(1.0, self.step_count / max(1, self.warmup_steps))
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        # Cosine annealing phase
        else:
            # Check for cycle restart BEFORE incrementing
            if self.cycle_step >= self.current_cycle_length:
                self.cycle_step = 0
                self.cycle_count += 1
                self.current_cycle_length = max(1, int(self.current_cycle_length * self.restart_mult))
            
            # Calculate progress within current cycle
            progress = self.cycle_step / max(1, self.current_cycle_length)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            
            # Increment AFTER calculation
            self.cycle_step += 1
        
        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]