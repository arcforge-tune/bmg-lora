from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
import datasets

def preprocess_gpt2_instruction_lm(example, tokenizer, max_length):
    if "instruction" not in example or "input" not in example or "output" not in example:
        raise ValueError(f"Missing required field in example: {example}")
    text = f"{example['instruction']}\n{example['input']}"
    label = example["output"]
    input_enc = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    label_enc = tokenizer(label, padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": label_enc["input_ids"]
    }

def load_dataset(data_config, model_config):
    # Support both hub datasets (with split) and local files
    if 'dataset_name' in data_config:
        dataset = datasets.load_dataset(
            data_config['dataset_name'],
            split=data_config['split']
        )
    else:
        dataset = hf_load_dataset(
            data_config['dataset_type'],
            data_files=data_config['data_files']
        )["train"]
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer_id'])
    if model_config.get('pad_token_as_eos', False):
        tokenizer.pad_token = tokenizer.eos_token
    model_id = model_config.get('model_id', '').lower()
    if model_id == 'gpt2':
        preprocess_fn = lambda ex: preprocess_gpt2_instruction_lm(ex, tokenizer, data_config['max_length'])
    elif 'custom_preprocess_fn' in data_config and callable(data_config['custom_preprocess_fn']):
        preprocess_fn = data_config['custom_preprocess_fn']
    else:
        raise ValueError(f"No preprocessing function available for model_id '{model_id}'. Please provide a custom_preprocess_fn in data_config.")
    # Preprocess and tokenize
    tokenized = dataset.map(preprocess_fn, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # Split (only if not using a split from hub)
    if 'dataset_name' in data_config and 'split_ratio' not in data_config:
        train_loader = DataLoader(tokenized, batch_size=data_config['batch_size'], shuffle=True)
        val_loader = None
    else:
        split_ratio = data_config['split_ratio']
        train_size = int(split_ratio * len(tokenized))
        val_size = len(tokenized) - train_size
        train_data, val_data = random_split(tokenized, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=data_config['batch_size'], shuffle=True)
        val_loader = DataLoader(val_data, batch_size=data_config['batch_size'])
    return train_loader, val_loader