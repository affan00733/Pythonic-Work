import torch
import json
import datetime
import re
import os
import logging
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM, pipeline

torch.cuda.empty_cache()
torch.cuda.synchronize()

CACHE_PATH = "/fs/clip-scratch/masumd1/.cache/huggingface"
os.environ['HF_HOME'] = CACHE_PATH
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH + "/models"
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH + "/datasets"
os.environ['TORCH_HOME'] = "/fs/clip-scratch/masumd1/.cache/torch"

IMAGE_METADATA_PATH = "./data/CrisisFACTS/rawdata/image_metadata_mapping.json"
QUERY_BASE_PATH = "./data/CrisisFACTS/event.metadata/crisisfacts.events/"
IMAGE_BASE_PATH = "./data/CrisisFACTS/rawdata/twitter/"
HIGH_PRIORITY_IMAGE_PATH = "./data/Crisis_Fact_high_priority/"  # High-priority images directory
OUTPUT_FOLDER = "./output_high_priority/"
LOG_FOLDER = "./logs/"

os.makedirs(LOG_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

log_filename = f"process_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
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

# logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
# logging.basicConfig(filename=LOG_FILE, level=logging.warning, format="%(asctime)s - %(levelname)s - %(message)s")
# logging.basicConfig(filename=LOG_FILE, level=logging.error, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
SUMMARIZATION_MODEL_ID = "facebook/bart-large-cnn"  # A strong summarization model
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

summarizer = pipeline("summarization", model=SUMMARIZATION_MODEL_ID, device=0 if torch.cuda.is_available() else -1)


def convert_unix_to_date(unix_timestamp):
    """Convert UNIX timestamp to YYYY-MM-DD format."""
    return datetime.datetime.utcfromtimestamp(unix_timestamp).strftime('%Y-%m-%d')

def get_query_json_path(event_id, date_str):
    """Get the path for the queries JSON file based on event_id and date."""
    return os.path.join(QUERY_BASE_PATH, event_id, f"{date_str}.json")

def get_image_path(trecis_id, image_filename):
    """Get the path for the image using trecisID transformation."""
    trecis_new = trecis_id.replace("TRECIS-CTIT-H-", "TRECIS.")
    old_path = os.path.join(IMAGE_BASE_PATH, trecis_new, trecis_id)
    return os.path.join(old_path, image_filename)
def find_image_in_subfolders(image_filename):
    """
    Search for the image filename in any subfolder under HIGH_PRIORITY_IMAGE_PATH.
    """
    for root, _, files in os.walk(HIGH_PRIORITY_IMAGE_PATH):
        if image_filename in files:
            return os.path.join(root, image_filename)
    return None

def get_high_priority_image_list():
    """Get the list of high-priority image filenames from the high-priority folder."""
    high_priority_images = set()
    for folder in os.listdir(HIGH_PRIORITY_IMAGE_PATH):
        folder_path = os.path.join(HIGH_PRIORITY_IMAGE_PATH, folder)
        if os.path.isdir(folder_path):
            for image in os.listdir(folder_path):
                high_priority_images.add(image)  # Add only image filenames, no full paths
    logging.info(f"Found {len(high_priority_images)} high-priority images.")
    return high_priority_images




def extract_facts(image_path, event_type, queries):
    """Run the model on an image for multiple queries and return results."""
    try:
        image = Image.open(image_path).convert("RGB")
    except UnidentifiedImageError:
        logging.error(f"Unidentified image error: Cannot identify {image_path}. Skipping.")
        return None
    except Exception as e:
        logging.error(f"Error opening image {image_path}: {e}")
        return None

    results = {}
    for query_data in queries:
        query_text = query_data["query"]
        query_id = query_data["queryID"]

        prompt_text = f"For this {event_type} of the following image <|image_1|>, answer the following question: {query_text}. If no relevant information is found, respond with 'Not available'."


        messages = [{"role": "user", "content": prompt_text}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        inputs = processor(prompt, [image], return_tensors="pt").to(device)

        generation_args = {
            "max_new_tokens": 1024,
            "temperature": 0.0,
            "do_sample": False,
        }

        try:
            with torch.no_grad():
                generate_ids = model.generate(**inputs, eos_token_id=processor.tokenizer.eos_token_id,
                                              **generation_args)

            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            response_text = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

            results[query_id] = {
                "query": query_text,
                "prompt": prompt_text,
                "response": response_text
            }
        except Exception as e:
            logging.error(f"Error generating response for query {query_id} on image {image_path}: {e}")
            results[query_id] = {
                "query": query_text,
                "prompt": prompt_text,
                "response": "Error processing this query"
            }
    return results

import re

def generate_summary(responses):
    """Summarize useful information from extracted facts."""
    # logging.info(f"Generating summary for {len(responses)} queries.")
    torch.cuda.empty_cache()
    # Define a robust regex pattern to detect variations of "Not available"
    not_available_pattern = re.compile(r"^\s*not\s*available[.!]?\s*$", re.IGNORECASE)

    # Filter out responses that match the "Not available" pattern
    valid_facts = [
        resp["response"].strip()
        for resp in responses.values()
        if not not_available_pattern.match(resp["response"].strip())
    ]

    # logging.info(f"Found {len(valid_facts)} valid facts.")

    if not valid_facts:
        logging.warning("No valid facts found, returning default summary.")
        return "No useful information available."

    # Combine valid responses into a single text
    combined_text = " ".join(valid_facts)

    logging.info(f"Before summarization, combined_text: {combined_text}")

    # Limit input size to avoid memory issues
    # if len(combined_text.split()) > 1024:
    #     logging.warning("Trimming input text as it exceeds token limits.")
    #     combined_text = " ".join(combined_text.split()[:1000])  # Truncate text

    try:
        # Summarization
        summary = summarizer(combined_text, max_length=150, min_length=50, do_sample=False)[0]["summary_text"]
        logging.info(f"Generated summary: {summary}")
        return summary
    except Exception as e:
        logging.error(f"Summarization error: {e}")
        return "Error generating summary."


high_priority_images = set()

# Iterate over the subfolders inside HIGH_PRIORITY_IMAGE_PATH
for folder in os.listdir(HIGH_PRIORITY_IMAGE_PATH):
    folder_path = os.path.join(HIGH_PRIORITY_IMAGE_PATH, folder)

    # Ensure we only process directories, ignoring files like ".DS_Store"
    if os.path.isdir(folder_path):
        for img in os.listdir(folder_path):
            high_priority_images.add(img)  # Store only filenames

logging.info(f"Found {len(high_priority_images)} high-priority images.")
logging.info(f"Found {len(high_priority_images)} high-priority images.")
try:
    with open(IMAGE_METADATA_PATH, "r") as f:
        image_metadata = json.load(f)
except Exception as e:
    logging.error(f"Error loading image metadata: {e}")
    raise

filtered_metadata = [img for img in image_metadata if img["image_filename"] in high_priority_images]

processed_images = set(os.listdir(OUTPUT_FOLDER))

total_images = len(filtered_metadata)
already_processed = sum(1 for metadata in filtered_metadata if f"{metadata['image_filename']}.json" in processed_images)
remaining_images = total_images - already_processed

logging.info(f"Total Images: {total_images}")
logging.info(f"Already Processed: {already_processed}")
logging.info(f"Remaining to Process: {remaining_images}")

skipped_images = 0
processed_count = 0

for metadata in filtered_metadata:
    try:
        image_filename = metadata["image_filename"]
        unique_id = metadata["unique_id"]
        event_id = metadata["event_id"]
        unix_timestamp = metadata["unix_timestamp"]

        date_str = convert_unix_to_date(unix_timestamp)

        query_file_path = get_query_json_path(event_id, date_str)
        if not os.path.exists(query_file_path):
            logging.warning(f"Query file missing: {query_file_path}, skipping {image_filename}")
            continue
        with open(query_file_path, "r") as f:
            query_data = json.load(f)

        event_type = query_data["event"]["type"]
        trecis_id = query_data["event"]["trecisID"]
        queries = query_data["queries"]
        # image_path = get_image_path(trecis_id, image_filename)
        # image_path = os.path.join(HIGH_PRIORITY_IMAGE_PATH, trecis_id + ".high_pr", image_filename)
        image_path = find_image_in_subfolders(image_filename)
        if not image_path:
            logging.warning(f"Image file not found: {image_filename}, skipping.")
            continue

        output_file = os.path.join(OUTPUT_FOLDER, f"{image_filename}.json")
        if os.path.exists(output_file):
            skipped_images += 1
            continue

        if not os.path.exists(image_path):
            logging.warning(f"Image file missing: {image_path}, skipping {image_filename}")
            continue
        results = extract_facts(image_path, event_type, queries)
        logging.info(f"results {results}")
        summary = generate_summary(results)
        logging.info("HERE3")
        output_data = {
            "image_filename": image_filename,
            "queries": results,
            "summary": summary
        }
        logging.info("HERE4")
        with open(output_file, "w") as f:
            json.dump(output_data, f, indent=4)

        processed_count += 1
        logging.info(f"Processed: {image_filename} ({processed_count}/{remaining_images})")
    except Exception as e:
        logging.error(f"Unexpected error processing {image_filename}: {e}")

logging.info(f"Final Summary:")
logging.info(f"Total Images: {total_images}")
logging.info(f"Already Processed: {already_processed}")
logging.info(f"Remaining to Process: {remaining_images}")
logging.info(f"Skipped Images: {skipped_images}")
logging.info(f"Newly Processed Images: {processed_count}")