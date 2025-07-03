from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
import datasets

def preprocess_instruction_lm(example, tokenizer, max_length):
    text = f"{example['instruction']}\n{example['input']}"
    label = example["output"]

    input_enc = tokenizer(text, padding="max_length", truncation=True, max_length=max_length)
    label_enc = tokenizer(label, padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": label_enc["input_ids"]
    }

def preprocess_text_lm(example, tokenizer, max_length):
    # For plain language modeling (no instruction, just text)
    input_enc = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": input_enc["input_ids"]
    }

def preprocess_classification(example, tokenizer, max_length, label2id):
    input_enc = tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)
    label = label2id[example["label"]]
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": label
    }

def get_preprocess_fn(task_type, model_config, data_config):
    if task_type == "CAUSAL_LM_INSTRUCTION":
        return lambda ex: preprocess_instruction_lm(ex, tokenizer, data_config['max_length'])
    elif task_type == "CAUSAL_LM_TEXT":
        return lambda ex: preprocess_text_lm(ex, tokenizer, data_config['max_length'])
    elif task_type == "CLASSIFICATION":
        label2id = {l: i for i, l in enumerate(data_config.get('label_list', []))}
        return lambda ex: preprocess_classification(ex, tokenizer, data_config['max_length'], label2id)
    else:
        raise ValueError(f"Unknown task_type: {task_type}")

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
    task_type = model_config['lora'].get('task_type', 'CAUSAL_LM_INSTRUCTION')
    preprocess_fn = get_preprocess_fn(task_type, model_config, data_config)
    # Preprocess and tokenize
    tokenized = dataset.map(preprocess_fn, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # Convert to TensorDataset for compatibility if needed

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