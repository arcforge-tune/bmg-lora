import argparse
import os
import yaml
from data.dataset_loader import load_dataset
from models.model_factory import create_model
from training.trainer import Trainer

def main():
# Set up argument parser
    parser = argparse.ArgumentParser(description='Load the LoRa configuration')
    parser.add_argument('--config', type=str, required=False,
                       help='Read the configuration from the default folder',
                       default="src/config/llama3.18B_qlora_config.yaml")
    parser.add_argument("--resume", help="Path to checkpoint directory", default=None)
    parser.add_argument("--skip_last_n_epochs", type=int, default=0,
                       help="Number of final epochs to skip (default: 0)")
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

    # Load dataset
    tokenizer, train_data, val_data = load_dataset(config)

    # Create model
    model, device = create_model(config, checkpoint_path=args.resume)

    # Initialize trainer
    trainer = Trainer(model, device, train_data, val_data, config, tokenizer)

    # Start training
    trainer.train(use_amp=True, use_tqdm=True, resume_checkpoint=args.resume,
                 skip_last_n_epochs=args.skip_last_n_epochs)

if __name__ == "__main__":
    main()