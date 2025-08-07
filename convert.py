import csv
import json

# Paths
input_tsv = "output/predictions/picard_dev_predictions.tsv"
output_json = "output/predictions/t5_dev_predictions_picard_spider.json"
dev_json_path = "data/spider_data/dev.json"

# Load DB IDs from Spider's dev.json
with open(dev_json_path, "r", encoding="utf-8") as f:
    dev_data = json.load(f)
    db_ids = [ex["db_id"] for ex in dev_data]

# Convert TSV to JSON
results = []
with open(input_tsv, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for idx, row in enumerate(reader):
        results.append({
            "query": row["predicted_sql"],
            "gold": row["ground_truth_sql"],
            "db_id": db_ids[idx],
            "exact_match": False  # placeholder, will be computed
        })

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print(f"✅ Saved predictions in Spider format to {output_json}")
