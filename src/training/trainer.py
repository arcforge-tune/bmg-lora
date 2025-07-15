import os
import torch
import math
from torch.optim import AdamW
from tqdm import tqdm
from peft import get_peft_model_state_dict

class ModelInstructions:
    def batch_to_device(self, batch, device):
        return {k: v.to(device) for k, v in batch.items()}

class CustomInstructions(ModelInstructions):
    pass

class Trainer:
    def __init__(self, model, device, train_loader, val_loader, config, tokenizer=None, instructions=None):
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
        if 'scheduler' in configTrain:
            self.scheduler = CosineAnnealingWarmRestScheduler(self.optimizer, configTrain)
        else:
            self.scheduler = None
        self.max_grad_norm = float(configTrain.get('max_grad_norm', 1.0))
        self.output_dir = configTrain.get('output_dir', 'outputs/lora_finetuned')
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Calculate fixed total steps
        self.batches_per_epoch = len(self.train_loader)
        self.grad_accum_steps = self.configTrain.get('gradient_accumulation_steps', 1)
        self.epochs = configTrain['epochs']
        self.total_steps = (self.epochs * self.batches_per_epoch) // self.grad_accum_steps

    def train(self, use_amp=False, save_checkpoint_fn=None, use_tqdm=False, resume_checkpoint=None):
        # Initialize training state
        start_epoch = 0
        global_step = 0
        batches_done_in_epoch = 0

        # Resume from checkpoint if provided
        if resume_checkpoint:
            start_epoch, global_step = self.load_checkpoint(resume_checkpoint)
            batches_done_in_epoch = (global_step * self.grad_accum_steps) % self.batches_per_epoch
            print(f"\nResuming: epoch={start_epoch}, global_step={global_step}, "
                  f"batches_done={batches_done_in_epoch}/{self.batches_per_epoch}, "
                  f"total_steps={self.total_steps}")

        for epoch in range(start_epoch, self.epochs):
            self.model.train()
            total_loss = 0.0
            self.optimizer.zero_grad()
            
            # Prepare data iterator
            if epoch == start_epoch and resume_checkpoint:
                data_iterator = self._fast_forward_data_loader(global_step)
                remaining_batches = self.batches_per_epoch - batches_done_in_epoch
            else:
                data_iterator = iter(self.train_loader)
                batches_done_in_epoch = 0
                remaining_batches = self.batches_per_epoch

            # Setup progress bar
            if use_tqdm:
                progress = tqdm(
                    data_iterator,
                    desc=f"Epoch {epoch+1}/{self.epochs}",
                    total=remaining_batches,
                    initial=batches_done_in_epoch
                )
            else:
                progress = data_iterator

            for step_offset, batch in enumerate(progress, start=1):
                current_batch = batches_done_in_epoch + step_offset
                
                # Move batch to device
                batch = self.instructions.batch_to_device(batch, self.device)
                
                # Forward pass
                if use_amp:
                    with torch.xpu.amp.autocast(dtype=torch.bfloat16):
                        outputs = self.model(**batch)
                        loss = outputs.loss
                        scaled_loss = loss / self.grad_accum_steps
                    scaled_loss.backward()
                else:
                    outputs = self.model(**batch)
                    loss = outputs.loss
                    (loss / self.grad_accum_steps).backward()
                    
                total_loss += loss.item()
                
                # Optimization step at accumulation boundary
                if current_batch % self.grad_accum_steps == 0:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                    
                    # Optimizer step
                    self.optimizer.step()
                    
                    # Scheduler step
                    if self.scheduler:
                        current_lr = self.scheduler.step()
                    else:
                        current_lr = self.optimizer.param_groups[0]['lr']
                    
                    self.optimizer.zero_grad()
                    global_step += 1
                    
                    # Save checkpoint
                    save_steps = self.configTrain.get('save_steps', 100)
                    if save_steps and global_step % save_steps == 0:
                        if save_checkpoint_fn and callable(save_checkpoint_fn):
                            save_checkpoint_fn()
                        else:
                            self.save_checkpoint(global_step, epoch)
                        print(f"Scheduler state: step_count={self.scheduler.step_count}, "
                              f"cycle_step={self.scheduler.cycle_step}")
                
                # Update progress bar (FIXED: Use fixed total_steps)
                if use_tqdm:
                    avg_loss = total_loss / current_batch
                    current_lr = self.optimizer.param_groups[0]['lr']
                    progress.set_postfix({
                        'loss': f'{avg_loss:.4f}',
                        'lr': f'{current_lr:.2e}',
                        'step': f'{global_step}/{self.total_steps}'  # Fixed total
                    })
            
            # Reset batch counter after epoch
            batches_done_in_epoch = 0
            
            # Calculate epoch metrics
            epoch_loss = total_loss / self.batches_per_epoch
            print(f"Epoch {epoch+1}/{self.epochs} completed - Avg Loss: {epoch_loss:.4f}")
            
            # Validation
            if self.val_loader:
                self.validate(epoch+1, use_amp)
                
            # Clear cache
            torch.xpu.empty_cache()
        
        # Save final model
        if self.configTrain.get('save_model', True):
            print("\n[XPU] Saving final LoRA adapter...")
            output_dir = self.configTrain.get('output_dir', 'outputs/lora_finetuned')
            os.makedirs(output_dir, exist_ok=True)
            self.model.save_pretrained(output_dir)
            self.tokenizer.save_pretrained(output_dir)
            print(f"\n[XPU] Final model saved to {output_dir}")

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

    def save_checkpoint(self, global_step, epoch):
        checkpoint_dir = os.path.join(
            self.output_dir, 
            f"checkpoint-epoch{epoch+1}-step{global_step}"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save ONLY adapter weights (not full model)
        if hasattr(self.model, 'save_adapter'):
            self.model.save_adapter(checkpoint_dir, "lora_adapter")
        elif hasattr(self.model, 'peft_config'):  # For newer PEFT versions
            self.model.save_pretrained(checkpoint_dir)
        
        # Save tokenizer
        if self.tokenizer:
            self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save training state separately
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'optimizer_state': self.optimizer.state_dict(),
            'scheduler_state': self.scheduler.state_dict() if self.scheduler else None,
            'config': self.configTrain
        }
        torch.save(checkpoint, os.path.join(checkpoint_dir, "training_state.pt"))
        
        # Save adapter separately
        if hasattr(self.model, 'save_pretrained'):
            adapter_path = os.path.join(checkpoint_dir, "adapter_model.bin")
            torch.save(get_peft_model_state_dict(self.model), adapter_path)
        
        print(f"Checkpoint saved: Epoch {epoch+1}, Step {global_step} -> {checkpoint_dir}")
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_path):
        state_path = os.path.join(checkpoint_path, "training_state.pt")
        checkpoint = torch.load(state_path)
        
        # Load ONLY training state (not model weights)
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        if self.scheduler and checkpoint['scheduler_state']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state'])
            self.scheduler.step_count = checkpoint['scheduler_state']['step_count']
        
        return checkpoint['epoch'], checkpoint['global_step']

    def _fast_forward_data_loader(self, global_step):
        batches_to_skip = (global_step * self.grad_accum_steps) % self.batches_per_epoch
        print(f"Fast-forwarding data loader: skipping {batches_to_skip} batches")
        
        data_iterator = iter(self.train_loader)
        for _ in range(batches_to_skip):
            try:
                next(data_iterator)
            except StopIteration:
                # Restart from beginning if we hit dataset end
                data_iterator = iter(self.train_loader)
                next(data_iterator)
        
        return data_iterator

class CosineAnnealingWarmRestScheduler:
    def __init__(self, optimizer, configTrain):
        self.optimizer = optimizer
        scheduler_config = configTrain['scheduler']
        
        # Validate parameters
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
        
        # State dict methods
        self.state_dict = self._state_dict
        self.load_state_dict = self._load_state_dict

    def step(self):
        self.step_count += 1
        
        if self.step_count <= self.warmup_steps:
            # Linear warmup
            progress = min(1.0, self.step_count / max(1, self.warmup_steps))
            lr = self.min_lr + (self.max_lr - self.min_lr) * progress
        else:
            # Cycle restart check
            if self.cycle_step >= self.current_cycle_length:
                self.cycle_step = 0
                self.cycle_count += 1
                self.current_cycle_length = max(1, int(self.current_cycle_length * self.restart_mult))
            
            # Cosine calculation
            progress = self.cycle_step / max(1, self.current_cycle_length)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
            
            # Increment after calculation
            self.cycle_step += 1
        
        # Update optimizer LR
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
            
        return lr
    
    def get_last_lr(self):
        return [param_group['lr'] for param_group in self.optimizer.param_groups]
    
    def _state_dict(self):
        return {
            'step_count': self.step_count,
            'cycle_step': self.cycle_step,
            'current_cycle_length': self.current_cycle_length,
            'cycle_count': self.cycle_count,
            'min_lr': self.min_lr,
            'max_lr': self.max_lr,
            'warmup_steps': self.warmup_steps,
            'restart_interval': self.restart_interval,
            'restart_mult': self.restart_mult
        }
    
    def _load_state_dict(self, state_dict):
        self.step_count = state_dict['step_count']
        self.cycle_step = state_dict['cycle_step']
        self.current_cycle_length = state_dict['current_cycle_length']
        self.cycle_count = state_dict['cycle_count']
        self.min_lr = state_dict.get('min_lr', self.min_lr)
        self.max_lr = state_dict.get('max_lr', self.max_lr)
        self.warmup_steps = state_dict.get('warmup_steps', self.warmup_steps)
        self.restart_interval = state_dict.get('restart_interval', self.restart_interval)
        self.restart_mult = state_dict.get('restart_mult', self.restart_mult)