import torch
import json
import datetime
import re
import os
import logging
import pandas as pd
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM

CACHE_PATH = "/fs/clip-scratch/masumd1/.cache/huggingface"
os.environ['HF_HOME'] = CACHE_PATH
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH + "/models"
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH + "/datasets"
os.environ['TORCH_HOME'] = "/fs/clip-scratch/masumd1/.cache/torch"

IMAGE_METADATA_PATH = "./data/CrisisFACTS/rawdata/image_metadata_mapping.json"
QUERY_BASE_PATH = "./data/CrisisFACTS/event.metadata/crisisfacts.events/"
IMAGE_BASE_PATH = "./data/CrisisFACTS/rawdata/twitter/"
OUTPUT_FOLDER = "./output/"
LOG_FOLDER = "./logs/"

log_filename = f"process_log_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
LOG_FILE = os.path.join(LOG_FOLDER, log_filename)

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
logging.basicConfig(filename=LOG_FILE, level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logging.basicConfig(filename=LOG_FILE, level=logging.warning, format="%(asctime)s - %(levelname)s - %(message)s")
logging.basicConfig(filename=LOG_FILE, level=logging.error, format="%(asctime)s - %(levelname)s - %(message)s")

MODEL_ID = "microsoft/Phi-3.5-vision-instruct"
device = "cuda" if torch.cuda.is_available() else "cpu"

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2')
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

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

        prompt_text = f"For this {event_type} of the following image <|image_1|>, answer the following question: {query_text}"

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


try:
    with open(IMAGE_METADATA_PATH, "r") as f:
        image_metadata = json.load(f)
except Exception as e:
    logging.error(f"Error loading image metadata: {e}")
    raise

processed_images = set(os.listdir(OUTPUT_FOLDER))

total_images = len(image_metadata)
already_processed = sum(1 for metadata in image_metadata if f"{metadata['image_filename']}.json" in processed_images)
remaining_images = total_images - already_processed

logging.info(f"Total Images: {total_images}")
logging.info(f"Already Processed: {already_processed}")
logging.info(f"Remaining to Process: {remaining_images}")

skipped_images = 0
processed_count = 0

for metadata in image_metadata:
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
        image_path = get_image_path(trecis_id, image_filename)

        output_file = os.path.join(OUTPUT_FOLDER, f"{image_filename}.json")
        if os.path.exists(output_file):
            skipped_images += 1
            continue

        if not os.path.exists(image_path):
            logging.warning(f"Image file missing: {image_path}, skipping {image_filename}")
            continue

        results = extract_facts(image_path, event_type, queries)

        output_data = {
            "image_filename": image_filename,
            "queries": results
        }

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