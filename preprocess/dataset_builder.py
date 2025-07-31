import json
import csv
from typing import List, Tuple
from schema_utils import load_tables, flatten_schema


def load_spider_data(path: str) -> List[dict]:
    """Load Spider dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_t5_input_output_pairs(examples: List[dict], schemas: dict) -> List[Tuple[str, str]]:
    """Convert Spider examples into (input, target) pairs for T5."""
    pairs = []
    for ex in examples:
        question = ex["question"].strip()
        db_id = ex["db_id"]
        sql = ex["query"].strip()  # raw SQL string

        # Flatten schema for the current db
        schema = flatten_schema(schemas[db_id]).strip()

        # Build input and output
        input_text = f"question: {question} schema: {schema}"
        target_text = sql

        pairs.append((input_text, target_text))
    return pairs


def save_to_tsv(pairs: List[Tuple[str, str]], path: str):
    """Save (input, target) pairs to a TSV file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["input", "target"])
        writer.writerows(pairs)


if __name__ == "__main__":
    # Paths based on your folder structure
    schema_path = "data/spider_data/tables.json"
    train_path = "data/spider_data/train_spider.json"
    output_path = "output/processed/train.tsv"

    # Ensure directory exists
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load data
    schemas = load_tables(schema_path)
    examples = load_spider_data(train_path)

    # Convert
    pairs = build_t5_input_output_pairs(examples, schemas)

    # Save or print
    print("Sample:")
    print("Input: ", pairs[0][0])
    print("Target:", pairs[0][1])
    
    save_to_tsv(pairs, output_path)
    print(f"Saved {len(pairs)} examples to {output_path}")
