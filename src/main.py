import os
import yaml
from data.dataset_loader import load_dataset
from models.model_factory import create_model
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.warning_filter import WarningFilter

WarningFilter.suppress()

def main():
    # Load configuration
    config_path = os.path.join(os.path.dirname(__file__), 'config', 'gpt2_lora_finetune_config.yaml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    # Apply environment variables from config
    if 'environment' in config:
        for key, value in config['environment'].items():
            os.environ[key] = str(value)

    # Setup logging
    setup_logger()

    # Load dataset
    train_data, val_data = load_dataset(config)

    # Create model
    model = create_model(config)

    # Initialize trainer
    trainer = Trainer(model, train_data, val_data, config)

    # Start training
    trainer.train(use_amp=True,use_tqdm=True)

if __name__ == "__main__":
    main()