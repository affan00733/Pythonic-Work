import os
import json
import logging
import glob
import torch
import pandas as pd
import bert_score
from collections import defaultdict

CACHE_PATH = "/fs/clip-scratch/masumd1/.cache/huggingface"
os.environ['HF_HOME'] = CACHE_PATH
os.environ['TRANSFORMERS_CACHE'] = CACHE_PATH + "/models"
os.environ['HF_DATASETS_CACHE'] = CACHE_PATH + "/datasets"
os.environ['TORCH_HOME'] = "/fs/clip-scratch/masumd1/.cache/torch"

# ==== CONFIGURATION ====
IMAGE_METADATA_PATH = "./data/CrisisFACTS/rawdata/image_metadata_mapping.json"
COLLAPSED_FACTS_PATH = "./data/collapsed-event-days/"
SUMMARIES_PATH = "./output_high_priority/"  # Use the new summaries dataset
EVAL_OUTPUT_FILE = "./evaluation_results_high_priority.json"
LOG_FILE = f"./evaluation_log_{pd.Timestamp.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"

BERT_THRESHOLD = 0.91

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
logging.info("Evaluation started on high-priority dataset.")

device = "cuda" if torch.cuda.is_available() else "cpu"

# ==== FILTER HIGH-PRIORITY DATASET ====
high_priority_images = set()
try:
    # Load high-priority images list from summaries
    for file_name in os.listdir(SUMMARIES_PATH):
        if file_name.endswith(".json"):
            high_priority_images.add(file_name.replace(".json", ""))
    logging.info(f"Loaded {len(high_priority_images)} high-priority image summaries.")
except Exception as e:
    logging.error(f"Error reading high-priority images dataset: {e}")

# ==== LOAD IMAGE METADATA (Filtered) ====
event_day_mapping = defaultdict(list)
try:
    with open(IMAGE_METADATA_PATH, "r") as f:
        image_metadata = json.load(f)

    for entry in image_metadata:
        image_filename = entry["image_filename"]
        if image_filename in high_priority_images:  # Filter only high-priority images
            event_id = entry["event_id"]
            r_tag = int(entry["r_tag"])
            event_day_mapping[(event_id, r_tag)].append(image_filename)

    logging.info(f"Filtered metadata contains {len(event_day_mapping)} event-day mappings.")

except Exception as e:
    logging.error(f"Error loading image metadata: {e}")

# ==== LOAD COLLAPSED FACTS (PER DAY) ====
collapsed_facts_by_day = defaultdict(list)
for file_path in glob.glob(os.path.join(COLLAPSED_FACTS_PATH, "Collapsed-*.json")):
    try:
        with open(file_path, "r") as f:
            collapsed_data = json.load(f)

        filename = os.path.basename(file_path).replace("Collapsed-", "").replace(".json", "")
        event_id, r_tag = filename.split("-r")
        r_tag = int(r_tag)
        r_tag = int(r_tag)

        collapsed_facts_by_day[(event_id, r_tag)].extend(fact["fact_text"] for fact in collapsed_data)

        logging.info(f"Loaded {len(collapsed_data)} collapsed facts for {event_id}, Day {r_tag}.")

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")

# ==== LOAD SUMMARIZED FACTS FROM HIGH-PRIORITY IMAGES ====
summarized_facts_by_day = defaultdict(list)
for file_path in glob.glob(os.path.join(SUMMARIES_PATH, "*.json")):
    try:
        with open(file_path, "r") as f:
            summary_data = json.load(f)

        image_filename = summary_data["image_filename"]

        matched = [(event_id, r_tag) for (event_id, r_tag), images in event_day_mapping.items() if
                   image_filename in images]
        if not matched:
            continue

        event_id, r_tag = matched[0]
        summary_text = summary_data.get("summary", "").strip()

        # 🔹 Ensure text is meaningful
        if not summary_text or summary_text.lower() in ["no useful information available.", "not available", "n/a"]:
            logging.warning(f"Ignoring empty/missing summary for {image_filename}.")
            continue
        summarized_facts_by_day[(event_id, r_tag)].append(summary_text)

        logging.info(f"Loaded summarized fact for {image_filename} (Event: {event_id}, Day: {r_tag}).")

    except Exception as e:
        logging.error(f"Error loading {file_path}: {e}")

# 🔹 DEBUGGING: Print keys to verify event-day mappings
logging.info(f"collapsed_facts_by_day keys: {list(collapsed_facts_by_day.keys())}")
logging.info(f"summarized_facts_by_day keys: {list(summarized_facts_by_day.keys())}")


# ==== EVALUATION (BERT SCORE) ====
evaluation_results_by_day = {}

for (event_id, r_tag), x_facts in collapsed_facts_by_day.items():
    y_facts = summarized_facts_by_day.get((event_id, r_tag), [])

    # 🔹 Debugging log to check key mismatches
    if not y_facts:
        logging.warning(f"No summarized facts found for {event_id}, Day {r_tag}. Expected key: ({event_id}, {r_tag})")
        continue

    try:
        logging.info(f"x_facts (Collapsed): {len(x_facts)} entries")
        logging.info(f"y_facts (Summarized): {len(y_facts)} entries")

        # Cross-match y_facts with all x_facts
        match_count = 0
        best_matches = []

        for y_idx, y_fact in enumerate(y_facts):
            best_score = 0.0
            best_match_x_fact = None
            matched_x_facts = []

            for x_idx, x_fact in enumerate(x_facts):
                P, R, F1 = bert_score.score([y_fact], [x_fact], model_type="microsoft/deberta-xlarge-mnli",
                                            device=device, batch_size=32)
                similarity_score = F1.item()
                logging.info(f"score: {similarity_score}")
                # Track best match for this y_fact
                if similarity_score > best_score:
                    best_score = similarity_score
                    best_match_x_fact = x_fact

                # Count matches above threshold
                if similarity_score >= BERT_THRESHOLD:
                    matched_x_facts.append((x_idx, similarity_score, x_fact))
                    match_count += 1
                logging.info(f"matched_x_facts: {matched_x_facts}")

            best_matches.append({
                "y_fact_index": y_idx,
                "y_fact": y_fact,
                "best_match_x_fact": best_match_x_fact,
                "best_match_score": best_score,
                "matched_x_facts": matched_x_facts  # All matched x_facts above threshold
            })
            logging.info(f"Best matches: {best_matches}")
        # Compute Metrics
        metric_1 = match_count / max(len(x_facts), 1)
        metric_2 = match_count / max(len(y_facts), 1)

        evaluation_results_by_day[f"{event_id}-r{r_tag}"] = {
            "collapsed_facts_count": len(x_facts),
            "summarized_facts_count": len(y_facts),
            "matches": match_count,
            "metric_1": metric_1,
            "metric_2": metric_2,
            "best_matches": best_matches  # Store best match for each y_fact
        }

        logging.info(f"Evaluated {event_id}, Day {r_tag}: Metric 1 = {metric_1:.2f}, Metric 2 = {metric_2:.2f}")

    except Exception as e:
        logging.error(f"Error computing BERTScore for {event_id}, Day {r_tag}: {e}")


# ==== SAVE RESULTS ====
try:
    with open(EVAL_OUTPUT_FILE, "w") as f:
        json.dump(evaluation_results_by_day, f, indent=4)
    logging.info(f"Evaluation results saved to {EVAL_OUTPUT_FILE}.")
except Exception as e:
    logging.error(f"Error saving evaluation results: {e}")

print("Evaluation completed. Results saved to:", EVAL_OUTPUT_FILE)
