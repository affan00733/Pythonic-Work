import os
PATH = "/cmlscratch/masumd1/.cache/huggingface"
# os.environ['HF_HOME'] = PATH
# os.environ['TRANSFORMERS_CACHE'] = PATH + "/models"
# os.environ['HF_DATASETS_CACHE'] = PATH + "/datasets"
# os.environ['TORCH_HOME'] = "/cmlscratch/masumd1/.cache/torch"
# os.environ['TRITON_CACHE_DIR'] = "/cmlscratch/masumd1/.cache/triton"
import json
import re
import torch
import datetime
import pandas as pd
import numpy as np
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
import logging
from collections import Counter

LOG_FOLDER = "./sbatch_logs/"
os.makedirs(LOG_FOLDER, exist_ok=True)
log_filename = f"eval_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE = os.path.join(LOG_FOLDER, log_filename)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)  # Capture all log levels
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.DEBUG)  # Save all logs (INFO, WARNING, ERROR)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)  # Print logs at INFO level and above
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

set_seed(1234)
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"


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
    return data


data_records = read_data("output")
dataset = Dataset.from_pandas(pd.DataFrame(data_records))


def create_message_column(row):
    messages = [
        {"content": f"{row['instruction']}\n Input: {row['input']}", "role": "user"},
        {"content": f"{row['output']}", "role": "assistant"}
    ]
    return {"messages": messages}

MODEL_DIR = "./phi-3-mini-LoRA__JSON_USED"
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

def format_dataset_chatml(row):
    return {"text": tokenizer.apply_chat_template(row["messages"], add_generation_prompt=False, tokenize=False)}

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'
logging.info(f"{attn_implementation}")

dataset_chatml = dataset.map(create_message_column)
dataset_chatml = dataset_chatml.map(format_dataset_chatml)
dataset_chatml = dataset_chatml.train_test_split(test_size=0.05, seed=1234)

for i in range(len(dataset_chatml['train'])):
    text = dataset_chatml['train'][i]['text']
    if '<|assistant|>' not in text or '<|user|>' not in text:
        logging.info(text)
        assert False
logging.info("verified")
logging.info(f"len train: {len(dataset_chatml['train'])}, len test: {len(dataset_chatml['test'])}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR, cache_dir=PATH + "/models", trust_remote_code=True, torch_dtype="auto", device_map="cuda", attn_implementation=attn_implementation
)

# Ensure correct padding side for Flash Attention compatibility
tokenizer.padding_side = "left"
# Evaluation and inference
def test_inference(prompt):
    # Prepare the prompt with chat template
    prompt = tokenizer.apply_chat_template([{
        "role": "user",
        "content": f"{prompt}"
    }], tokenize=False, add_generation_prompt=True)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    # Generate response
    outputs = model.generate(
        **inputs,
        max_new_tokens=4096
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=False)
    return result

def extract_json_from_text(text):
    """
    Extracts JSON from text enclosed within <|assistant|> and <|end|>
    and ensures valid parsing.
    """
    try:
        # Extract text between <|assistant|> and <|end|>
        match = re.search(r"<\|assistant\|>(.*?)<\|end\|>", text, re.DOTALL)
        if not match:
            return None  # No JSON found

        json_text = match.group(1).strip()  # Get extracted text
        return json.loads(json_text)  # Parse JSON safely

    except json.JSONDecodeError as e:
        print(f"JSON decoding error: {e}")
        return None


def compute_jaccard_similarity(list1, list2):
    """Computes Jaccard Similarity for two lists, ensuring all elements are hashable."""

    def convert_to_hashable(element):
        """Recursively convert elements into hashable types."""
        if isinstance(element, dict):
            return tuple(sorted((k, convert_to_hashable(v)) for k, v in element.items()))
        elif isinstance(element, list):
            return tuple(convert_to_hashable(e) for e in element)  # Convert list to tuple
        return element  # Return primitive types as-is

    # Convert all list elements to hashable types
    set1 = set(map(convert_to_hashable, list1))
    set2 = set(map(convert_to_hashable, list2))

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    return intersection / union if union > 0 else 0




def flatten_json(json_obj, prefix=""):
    """
    Recursively flattens a JSON dictionary into key-value pairs.
    Handles lists separately for partial matching.
    """
    items = []
    if isinstance(json_obj, dict):
        for key, value in json_obj.items():
            new_prefix = f"{prefix}.{key}" if prefix else key
            items.extend(flatten_json(value, new_prefix))
    elif isinstance(json_obj, list):
        items.append((prefix, json_obj))  # Store lists as lists (for Jaccard)
    else:
        items.append((prefix, str(json_obj)))  # Convert values to string
    return items

def compute_precision_recall(predicted, ground_truth):
    """
    Computes precision, recall, and F1-score for hierarchical JSON data.
    Handles lists using Jaccard Similarity.
    """
    # Parse JSON strings
    try:
        ground_truth_dict = json.loads(ground_truth)
        predicted_dict = predicted if isinstance(predicted, dict) else json.loads(predicted)
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON: {e}")
        return None

    # Flatten JSON structures
    gt_items = dict(flatten_json(ground_truth_dict))
    pred_items = dict(flatten_json(predicted_dict))

    # Compute per-key precision and recall
    key_results = {}
    tp_total, fp_total, fn_total = 0, 0, 0

    all_keys = gt_items.keys() | pred_items.keys()
    for key in all_keys:
        gt_value = gt_items.get(key, [])
        pred_value = pred_items.get(key, [])

        if isinstance(gt_value, list) and isinstance(pred_value, list):
            similarity = compute_jaccard_similarity(gt_value, pred_value)  # Uses fixed function
            tp = similarity * len(gt_value)  # Reward partial match
            fp = len(pred_value) - tp
            fn = len(gt_value) - tp

        else:
            # Exact match for non-list values
            tp = int(gt_value == pred_value)
            fp = int(pred_value is not None and gt_value != pred_value)
            fn = int(gt_value is not None and pred_value != gt_value)

        key_precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        key_recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        key_f1 = (2 * key_precision * key_recall) / (key_precision + key_recall) if (key_precision + key_recall) > 0 else 0

        key_results[key] = {
            "precision": key_precision,
            "recall": key_recall,
            "f1_score": key_f1
        }

        tp_total += tp
        fp_total += fp
        fn_total += fn

    # Compute overall precision, recall, and F1-score
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "overall": {
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score
        },
        "per_key": key_results
    }

def extracting_response(row, results_list):
    # logging.info(f"text passing {row}")
    response = test_inference(row['messages'][0]['content'])
    logging.info(f"before formatting response: {response}\n")
    logging.info(f"Output JSON: {json.dumps(row["output"], indent=2)}")
    processed_response = extract_json_from_text(response)
    if processed_response is None:
        logging.warning("Failed to extract JSON from response.")
        return None

    logging.info(f"Extracted JSON: {json.dumps(processed_response, indent=2)}")
    evaluation_results = compute_precision_recall(processed_response, row["output"])
    if evaluation_results:
        logging.info("Evaluation Results:")
        logging.info(f"Overall Metrics: {json.dumps(evaluation_results['overall'], indent=4)}")

        for key, scores in evaluation_results["per_key"].items():
            logging.info(
                f"Key: {key}, Precision: {scores['precision']:.4f}, Recall: {scores['recall']:.4f}, F1-score: {scores['f1_score']:.4f}")

        results_list.append({
            "input": row["messages"][0]["content"],
            "output": processed_response,
            "evaluation": evaluation_results
        })
EVALUATION_OUTPUT_FILE = "evaluation_results.json"

def export_results_to_json(results_list):
    """Exports the evaluation results to a JSON file."""
    with open(EVALUATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results_list, f, indent=4)
    logging.info(f"Evaluation results saved to {EVALUATION_OUTPUT_FILE}")

# List to store all evaluation results
evaluation_results_list = []

# Run evaluation
metricas = dataset_chatml['test'].select(range(0, 399)).map(
    lambda row: extracting_response(row, evaluation_results_list), batched=False
)

# Export the final evaluation results to a JSON file
export_results_to_json(evaluation_results_list)