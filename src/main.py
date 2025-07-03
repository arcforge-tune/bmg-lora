import os
import yaml
from data.dataset_loader import load_dataset
from models.model_factory import create_model
from training.trainer import Trainer
from utils.logger import setup_logger

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt2_lora_finetune_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Setup logging
    setup_logger()

    # Load dataset
    train_data, val_data = load_dataset(config['data'],config['model'])

    # Create model
    model = create_model(config['model'])

    # Initialize trainer
    trainer = Trainer(model, train_data, val_data, config['training'], config['model']['lora'])

    # Start training
    trainer.train()

if __name__ == "__main__":
    main()