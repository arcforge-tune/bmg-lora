import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, random_split, TensorDataset
from transformers import AutoTokenizer
import datasets

def preprocess_CLM_gpt2(example, tokenizer, max_length):
    prompt = f"{example['instruction']}\n{example['input']}"
    response = example["output"]

    # Do NOT add special tokens here (GPT-2 is decoder-only and has no [CLS], [SEP])
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
    response_enc = tokenizer(response, truncation=True, max_length=max_length, add_special_tokens=False)

    # Truncate response if needed
    available_length = max_length - len(prompt_enc["input_ids"])
    response_ids = response_enc["input_ids"][:available_length]

    input_ids = prompt_enc["input_ids"] + response_ids
    labels = [-100] * len(prompt_enc["input_ids"]) + response_ids
    attention_mask = [1] * len(input_ids)

    # Pad if needed
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def preprocess_CLM_llama(example, tokenizer, max_length):
    if "instruction" not in example or "input" not in example or "output" not in example:
        raise ValueError(f"Missing required field in example: {example}")
    instruction = example.get('instruction')
    input_context = example.get('input', '')
    assistant = example.get('output')

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_context}\n\n### Response:\n"
    response = f"\n{assistant}"

    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
    response_enc = tokenizer(response, truncation=True, max_length=max_length, add_special_tokens=False)

    # Truncate response if needed
    available_length = max_length - len(prompt_enc["input_ids"])
    response_ids = response_enc["input_ids"][:available_length]

    input_ids = prompt_enc["input_ids"] + response_ids
    labels = [-100] * len(prompt_enc["input_ids"]) + response_ids
    attention_mask = [1] * len(input_ids)

    # Pad if needed
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def preprocess_SFT_llama(example, tokenizer, max_length):
    if "instruction" not in example or "input" not in example or "output" not in example:
        raise ValueError(f"Missing required field in example: {example}")
    instruction = example.get('instruction')
    input_context = example.get('input', '')
    assistant = example.get('output')
    sys_prompt = example.get("system", "") or "You are a helpful assistant."
    # Combine instruction and input if input is present
    if input_context:
        user_message = f"{instruction}\n{input_context}"
    else:
        user_message = instruction

    prompt = f"<s>[INST] <<SYS>>\n{sys_prompt}\n<</SYS>>\n\n{user_message} [/INST] "
    response = f"{assistant}</s>"

    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
    response_enc = tokenizer(response, truncation=True, max_length=max_length, add_special_tokens=False)

    # Truncate response if needed
    available_length = max_length - len(prompt_enc["input_ids"])
    response_ids = response_enc["input_ids"][:available_length]

    input_ids = prompt_enc["input_ids"] + response_ids
    labels = [-100] * len(prompt_enc["input_ids"]) + response_ids
    attention_mask = [1] * len(input_ids)

    # Pad if needed
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        labels += [-100] * pad_len
        attention_mask += [0] * pad_len

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }

def preprocess_SFT_llama31(example, tokenizer, max_length):
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

    prompt = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n"
        f"{sys_prompt}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        f"{user_message}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>"
    )
    response = assistant + "<|eot_id|>"
    
    # Tokenize separately
    prompt_enc = tokenizer(prompt, truncation=True, max_length=max_length, add_special_tokens=False)
    response_enc = tokenizer(response, truncation=True, max_length=max_length, add_special_tokens=False)
    
    # Combine and create labels
    input_ids = prompt_enc["input_ids"] + response_enc["input_ids"]
    attention_mask = [1] * len(input_ids)


    # Ensure the combined length does not exceed max_length
    combined_length = len(prompt_enc["input_ids"]) + len(response_enc["input_ids"])
    if combined_length > max_length:
        # Adjust response length to fit within max_length
        response_enc = tokenizer(
            response,
            truncation=True,
            max_length=max_length - len(prompt_enc["input_ids"]),
            add_special_tokens=False,
        )
    
    labels = [-100] * len(prompt_enc["input_ids"]) + response_enc["input_ids"]
    
    # Pad labels to match max_length
    pad_len = max_length - len(labels)
    
    # Pad sequences
    pad_len = max_length - len(input_ids)
    if pad_len > 0:
        input_ids += [tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        labels += [-100] * pad_len
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
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
    tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer_id'], model_max_length=data_config['max_length'], padding_side="left", add_eos_token=False)
    # Explicitly set pad_token if not already defined
    if tokenizer.pad_token is None and model_config.get('pad_token_as_eos', False):
        tokenizer.pad_token = tokenizer.eos_token  # Fallback to eos_token if needed
        print(f"Warning: pad_token not set. Using eos_token ({tokenizer.eos_token}) as pad_token.")
    instruction_style = data_config.get('instruction_style', '')
    if instruction_style == 'CLM_gpt2':
        preprocess_fn = lambda ex: preprocess_CLM_gpt2(ex, tokenizer, data_config['max_length'])
    elif instruction_style == 'CLM_llama':
        preprocess_fn = lambda ex: preprocess_CLM_llama(ex, tokenizer, data_config['max_length'])
    elif instruction_style == 'SFT_llama':
        preprocess_fn = lambda ex: preprocess_SFT_llama(ex, tokenizer, data_config['max_length'])
    elif instruction_style == 'SFT_llama31':
        preprocess_fn = lambda ex: preprocess_SFT_llama31(ex, tokenizer, data_config['max_length'])
    elif custom_preprocess_fn is not None and callable(custom_preprocess_fn):
        preprocess_fn = custom_preprocess_fn
    else:
        raise ValueError(f"No preprocessing function set. Please provide a custom_preprocess_fn in data_config.")
    # Preprocess and tokenize
    # def debug_long_examples(example):
    #     if len(example["input_ids"]) > data_config["max_length"]:
    #         print(f"⚠️ Example too long: {len(example['input_ids'])} tokens")
    #     return example

    # tokenized = dataset.map(lambda ex: debug_long_examples(preprocess_fn(ex)), batched=False)
    tokenized = dataset.map(preprocess_fn, batched=False, remove_columns=dataset.column_names)
    tokenized = tokenized.filter(lambda row: len(row['input_ids']) <= data_config['max_length'])
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