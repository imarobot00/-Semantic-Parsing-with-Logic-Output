import json
import csv
from typing import List, Tuple
from schema_utils import load_tables, flatten_schema


def load_spider_data(path: str) -> List[dict]:
    """Load Spider dataset from JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_t5_input_output_pairs(examples: List[dict], schemas: dict) -> List[Tuple[str, str, str]]:
    """Convert Spider examples into (input, target, db_id) triples for T5."""
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

        pairs.append((input_text, target_text, db_id))
    return pairs


def save_to_tsv(pairs: List[Tuple[str, str, str]], path: str):
    """Save (input, target, db_id) triples to a TSV file."""
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["input", "target", "db_id"])
        writer.writerows(pairs)


if __name__ == "__main__":

    schema_path = "data/spider_data/tables.json"
    dev_path = "data/spider_data/dev.json"
    dev_output = "output/processed/dev.tsv"

    import os
    os.makedirs(os.path.dirname(dev_output), exist_ok=True)

    schemas = load_tables(schema_path)
    examples = load_spider_data(dev_path)

    pairs = build_t5_input_output_pairs(examples, schemas)

    print("Sample:")
    print("Input: ", pairs[0][0])
    print("Target:", pairs[0][1])
    
    save_to_tsv(pairs, dev_output)
    print(f"Saved {len(pairs)} examples to {dev_output}")

