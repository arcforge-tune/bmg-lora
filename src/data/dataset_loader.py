from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
import datasets

def preprocess_CLM_gpt2(example, tokenizer, max_length):
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

def preprocess_CLM_llama(example, tokenizer, max_length):
    if "instruction" not in example or "input" not in example or "output" not in example:
        raise ValueError(f"Missing required field in example: {example}")
    prompt = f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
    input_enc = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length)
    # For Llama2, use input_ids as both input and labels (causal LM)
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": input_enc["input_ids"]
    }

def preprocess_SFT_llama(example, tokenizer, max_length):
    if "instruction" not in example or "output" not in example:
        raise ValueError(f"Missing required fields in example: {example}")
    instruction = example.get("instruction", "")
    input_context = example.get("input", "")
    assistant = example.get("output", "")
    sys_prompt = example.get("system", "") or "You are a helpful assistant."
    # Combine instruction and input if input is present
    if input_context:
        user_message = f"{instruction}\n{input_context}"
    else:
        user_message = instruction

    prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_message} [/INST] {assistant}</s>"
    input_enc = tokenizer(prompt, padding="max_length", truncation=True, max_length=max_length)
    return {
        "input_ids": input_enc["input_ids"],
        "attention_mask": input_enc["attention_mask"],
        "labels": input_enc["input_ids"]
    }

#create a method load_dataset that receive config and call load_dataset(data_config, model_config)
def load_dataset(config):
    return load_dataset_config(config['data'],config['model'])

def load_dataset_config(data_config, model_config, custom_preprocess_fn = None):
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
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer_id'], trust_remote_code=True, model_max_length=data_config['max_length'],use_fast=False)
    if model_config.get('pad_token_as_eos', False):
        tokenizer.pad_token = tokenizer.eos_token
    instruction_style = data_config.get('instruction_style', '')
    if instruction_style == 'CLM_gpt2':
        preprocess_fn = lambda ex: preprocess_CLM_gpt2(ex, tokenizer, data_config['max_length'])
    elif instruction_style == 'CLM_llama':
        preprocess_fn = lambda ex: preprocess_CLM_llama(ex, tokenizer, data_config['max_length'])
    elif instruction_style == 'SFT_llama':
        preprocess_fn = lambda ex: preprocess_SFT_llama(ex, tokenizer, data_config['max_length'])
    elif custom_preprocess_fn is not None and callable(custom_preprocess_fn):
        preprocess_fn = custom_preprocess_fn
    else:
        raise ValueError(f"No preprocessing function set. Please provide a custom_preprocess_fn in data_config.")
    # Preprocess and tokenize
    tokenized = dataset.map(preprocess_fn, batched=False, remove_columns=dataset.column_names)
    tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    # Split (only if not using a split from hub)
    num_workers = data_config.get('num_workers', 0)
    if ('dataset_name' in data_config and 'split_ratio' not in data_config) or ('data_files' in data_config):
        train_loader = DataLoader(tokenized, batch_size=data_config['batch_size'], shuffle=True, num_workers=num_workers)
        val_loader = None
    else:
        split_ratio = data_config['split_ratio']
        train_size = int(split_ratio * len(tokenized))
        val_size = len(tokenized) - train_size
        train_data, val_data = random_split(tokenized, [train_size, val_size])
        train_loader = DataLoader(train_data, batch_size=data_config['batch_size'], shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_data, batch_size=data_config['batch_size'], num_workers=num_workers)
    return tokenizer, train_loader, val_loader