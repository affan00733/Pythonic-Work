import json
import os

# Input and output JSON file names
EVALUATION_INPUT_FILE = "evaluation_results.json"
FINAL_EVALUATION_OUTPUT_FILE = "final_evaluation.json"


def aggregate_evaluation_metrics():
    """
    Aggregates precision, recall, and F1-score across all evaluated samples.
    Computes total average for both overall and per-key metrics.
    """
    # Load evaluation results
    if not os.path.exists(EVALUATION_INPUT_FILE):
        print(f"Error: {EVALUATION_INPUT_FILE} not found.")
        return

    with open(EVALUATION_INPUT_FILE, "r", encoding="utf-8") as f:
        evaluation_data = json.load(f)

    # Initialize accumulators
    overall_metrics = {"precision": 0, "recall": 0, "f1_score": 0, "count": 0}
    per_key_metrics = {}
    column_color_mapping_metrics = {"precision": 0, "recall": 0, "f1_score": 0,
                                    "count": 0}  # Separate accumulator for column_color_mapping

    for entry in evaluation_data:
        if "evaluation" not in entry:
            continue

        eval_results = entry["evaluation"]

        # Aggregate overall metrics
        overall_metrics["precision"] += eval_results["overall"]["precision"]
        overall_metrics["recall"] += eval_results["overall"]["recall"]
        overall_metrics["f1_score"] += eval_results["overall"]["f1_score"]
        overall_metrics["count"] += 1

        # Aggregate per-key metrics
        for key, scores in eval_results["per_key"].items():
            # Handle column_color_mapping separately
            if key.startswith("chart_analysis.column_color_mapping."):
                column_color_mapping_metrics["precision"] += scores["precision"]
                column_color_mapping_metrics["recall"] += scores["recall"]
                column_color_mapping_metrics["f1_score"] += scores["f1_score"]
                column_color_mapping_metrics["count"] += 1
                continue  # Skip adding individual columns to per_key_metrics

            if key not in per_key_metrics:
                per_key_metrics[key] = {"precision": 0, "recall": 0, "f1_score": 0, "count": 0}

            per_key_metrics[key]["precision"] += scores["precision"]
            per_key_metrics[key]["recall"] += scores["recall"]
            per_key_metrics[key]["f1_score"] += scores["f1_score"]
            per_key_metrics[key]["count"] += 1

    # Compute final averages
    final_overall = {
        "precision": overall_metrics["precision"] / overall_metrics["count"] if overall_metrics["count"] > 0 else 0,
        "recall": overall_metrics["recall"] / overall_metrics["count"] if overall_metrics["count"] > 0 else 0,
        "f1_score": overall_metrics["f1_score"] / overall_metrics["count"] if overall_metrics["count"] > 0 else 0
    }

    final_per_key = {}
    for key, metrics in per_key_metrics.items():
        final_per_key[key] = {
            "precision": metrics["precision"] / metrics["count"] if metrics["count"] > 0 else 0,
            "recall": metrics["recall"] / metrics["count"] if metrics["count"] > 0 else 0,
            "f1_score": metrics["f1_score"] / metrics["count"] if metrics["count"] > 0 else 0
        }

    # Compute final scores for column_color_mapping
    if column_color_mapping_metrics["count"] > 0:
        final_per_key["chart_analysis.column_color_mapping"] = {
            "precision": column_color_mapping_metrics["precision"] / column_color_mapping_metrics["count"],
            "recall": column_color_mapping_metrics["recall"] / column_color_mapping_metrics["count"],
            "f1_score": column_color_mapping_metrics["f1_score"] / column_color_mapping_metrics["count"]
        }

    # Final aggregated results
    final_results = {
        "total_samples": overall_metrics["count"],
        "overall_evaluation": final_overall,
        "per_key_evaluation": final_per_key
    }

    # Save results to JSON file
    with open(FINAL_EVALUATION_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=4)

    print(f"Final evaluation saved to {FINAL_EVALUATION_OUTPUT_FILE}")


if __name__ == "__main__":
    aggregate_evaluation_metrics()
