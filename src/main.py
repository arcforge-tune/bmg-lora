import argparse
import os
import yaml
from data.dataset_loader import load_dataset
from models.model_factory import create_model
from training.trainer import Trainer
from utils.logger import setup_logger
from utils.warning_filter import WarningFilter

WarningFilter.suppress()

def main():
# Set up argument parser
    parser = argparse.ArgumentParser(description='Load the LoRa configuration')
    parser.add_argument('--config', type=str, required=False,
                       help='Read the configuration from the default folder',
                       default="llamma2_chat_hf_qlora_xpu_config.yaml")
    args = parser.parse_args()
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found")
        return
    
    # Load configuration
    with open(args.config, 'r') as file:
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
    model, device = create_model(config)

    # Initialize trainer
    trainer = Trainer(model, device, train_data, val_data, config)

    # Start training
    trainer.train(use_amp=True,use_tqdm=True)

if __name__ == "__main__":
    main()