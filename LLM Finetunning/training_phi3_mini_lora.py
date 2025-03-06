import os
PATH = "/cmlscratch/masumd1/.cache/huggingface"
# os.environ['HF_HOME'] = PATH
# os.environ['TRANSFORMERS_CACHE'] = PATH + "/models"
# os.environ['HF_DATASETS_CACHE'] = PATH + "/datasets"
# os.environ['TORCH_HOME'] = "/cmlscratch/masumd1/.cache/torch"
# os.environ['TRITON_CACHE_DIR'] = "/cmlscratch/masumd1/.cache/triton"
import json
import torch
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments, set_seed, pipeline
from peft import LoraConfig, prepare_model_for_kbit_training, TaskType, PeftModel
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
import wandb
wandb.login(key="a9085471c4efe63a7d4cbeef9e1d393c30a50e86")

import logging
from datetime import datetime

# Generate a unique log filename with timestamp
log_filename = f"sbatch_logs/training_phi3_mini_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Configure logging to create a new log file for every execution
logging.basicConfig(
    filename=log_filename,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.basicConfig(
    filename=log_filename,
    level=logging.warning,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logging.basicConfig(
    filename=log_filename,
    level=logging.error,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

# Environment variables for cache optimization


# Global parameters
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"
NEW_MODEL_NAME = "Phi-3-mini-finetuned"

set_seed(1234)


def read_data(output_folder):
    """Reads and processes dataset from the output folder."""
    data = []
    valid_count = 0
    skipped_folders = 0
    json_errors = 0
    for case_folder in os.listdir(output_folder):
        case_path = os.path.join(output_folder, case_folder)
        if not os.path.isdir(case_path):
            continue

        input_file = os.path.join(case_path, "data_gen.txt")
        chart_file = os.path.join(case_path, "chart_attributes.json")
        color_file = os.path.join(case_path, "color_mapping.json")

        if os.path.exists(input_file) and os.path.exists(chart_file):
            with open(input_file, "r", encoding="utf-8") as f:
                input_text = f.read()

            try:
                with open(chart_file, "r", encoding="utf-8") as f:
                    chart_content = f.read().strip()
                    if not chart_content:
                        raise ValueError("Empty JSON file")
                    chart_json = json.loads(chart_content)
            except (json.JSONDecodeError, ValueError) as e:
                json_errors += 1
                logging.warning(f"Warning: Skipping {chart_file} due to JSON error: {e}")
                continue  # Skip this case due to invalid JSON

            if os.path.exists(color_file):
                with open(color_file, "r", encoding="utf-8") as f:
                    color_mapping = json.load(f)
            else:
                color_mapping = {}  # Default to empty mapping if file is missing

            # Add color mapping to chart attributes``
            chart_json["chart_analysis"]["column_color_mapping"] = color_mapping

            data.append(
                {"instruction": "Generate a json for chart attributes", "input": input_text, "output": json.dumps(chart_json)})
            valid_count += 1
        else:
            skipped_folders += 1
    # logging.INFO(f"Total valid datasets with chart_attributes.json: {valid_count}")
    # logging.INFO(f"Total skipped subfolders without chart_attributes.json: {skipped_folders}")
    # logging.INFO(f"Total JSON errors encountered: {json_errors}")
    return data


# Load and preprocess data
data_records = read_data("output")
dataset = Dataset.from_pandas(pd.DataFrame(data_records))


def create_message_column(row):
    messages = [
        {"content": f"{row['instruction']}\n Input: {row['input']}", "role": "user"},
        {"content": f"{row['output']}", "role": "assistant"}
    ]
    return {"messages": messages}

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=PATH + "/models", trust_remote_code=False, add_eos_token=False, use_fast=True)
tokenizer.pad_token = tokenizer.unk_token
tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
tokenizer.padding_side = "left"

def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}


# This code block is used to set the compute data type and attention implementation based on whether bfloat16 is supported on the current CUDA device.

# 'torch.cuda.is_bf16_supported()' is a function that checks if bfloat16 is supported on the current CUDA device.
# If bfloat16 is supported, 'compute_dtype' is set to 'torch.bfloat16' and 'attn_implementation' is set to 'flash_attention_2'.
if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
# If bfloat16 is not supported, 'compute_dtype' is set to 'torch.float16' and 'attn_implementation' is set to 'sdpa'.
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'

# This line of code is used to print the value of 'attn_implementation', which indicates the chosen attention implementation.
# logging.INFO(attn_implementation)


# Load tokenizer and process dataset


dataset_chatml = dataset.map(create_message_column)
dataset_chatml = dataset_chatml.map(format_dataset_chatml)
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

for i in range(len(dataset_chatml['train'])):
    text = dataset_chatml['train'][i]['text']
    if '<|assistant|>' not in text or '<|user|>' not in text:
        logging.INFO(text)
        assert False

# logging.INFO("verified")

# Load model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, cache_dir=PATH + "/models", torch_dtype=compute_dtype, trust_remote_code=False, device_map="auto",
    attn_implementation=attn_implementation

)

# LoRA configuration
peft_config = LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.05, task_type=TaskType.CAUSAL_LM,
    target_modules=['k_proj', 'q_proj', 'v_proj', 'o_proj', "gate_proj", "down_proj", "up_proj"]
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./phi-3-mini-LoRA__JSON_USED",
    evaluation_strategy="steps",
    do_eval=True,
    optim="adamw_torch",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    per_device_eval_batch_size=1,
    log_level="debug",
    save_strategy="epoch",
    logging_steps=100,
    learning_rate=1e-4,
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    eval_steps=2000,
    num_train_epochs=2,
    warmup_ratio=0.1,
    lr_scheduler_type="linear",
    report_to="wandb",
    seed=42
)

# def formatting_prompts_func(example):
#     output_texts = []
#     for i in range(len(example['messages'])):
#         text = tokenizer.apply_chat_template(example['messages'][i], add_generation_prompt=False, tokenize=False)
#         output_texts.append(text)
#     return output_texts

response_template = '<|assistant|>'
collator = DataCollatorForCompletionOnlyLM(
    response_template=response_template,
    tokenizer=tokenizer
)

# Trainer
trainer = SFTTrainer(
    model=model, train_dataset=dataset_chatml["train"], eval_dataset=dataset_chatml["test"],
    peft_config=peft_config, tokenizer=tokenizer, args=training_args, data_collator=collator,
    max_seq_length=4096, dataset_text_field="text"
)
# logging.INFO(f"len train: {len(dataset_chatml['train'])}, len test: {len(dataset_chatml['test'])}")
# Train model
trainer.train()
trainer.save_model()

# print("training here done")
# MODEL_DIR = "./phi-3-mini-LoRA"
# # Load the tokenizer and model from the local directory
# tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     MODEL_DIR, cache_dir=PATH + "/models", trust_remote_code=True, torch_dtype="auto", device_map="cuda"
# )
#
# # Ensure correct padding side for Flash Attention compatibility
# tokenizer.padding_side = "left"
# # Evaluation and inference
# def test_inference(prompt):
#     # Prepare the prompt with chat template
#     prompt = tokenizer.apply_chat_template([{
#         "role": "user",
#         "content": f"Generate a json for chart attributes from the following data:\n{prompt}"
#     }], tokenize=False, add_generation_prompt=True)
#
#     # Tokenize input
#     inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
#     # Generate response
#     outputs = model.generate(
#         **inputs,
#         max_new_tokens=4096
#     )
#     result = tokenizer.decode(outputs[0], skip_special_tokens=False)
#
#     # Log input/output for debugging
#     # print(f"Input prompt:\n{prompt}\n")
#     print(f"Generated output:\n{result}\n")
#
#     return result
#
# from evaluate import load
# rouge_metric = load("rouge")
#
# def calculate_rouge(row):
#     response = test_inference(row['messages'][0]['content'])
#
#     result = rouge_metric.compute(predictions=[response], references=[row['output']], use_stemmer=True)
#     result = {key: value * 100 for key, value in result.items()}
#     result['response'] = response
#     return result
# print(len(dataset_chatml['train']))
# metricas = dataset_chatml['test'].select(range(0, 400)).map(calculate_rouge, batched=False)
# import numpy as np
# print("Rouge 1 Mean: ", np.mean(metricas['rouge1']))
# print("Rouge 2 Mean: ", np.mean(metricas['rouge2']))
# print("Rouge L Mean: ", np.mean(metricas['rougeL']))
# print("Rouge Lsum Mean: ", np.mean(metricas['rougeLsum']))
