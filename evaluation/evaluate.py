import os
import csv
import torch
from tqdm import tqdm
from transformers import T5Tokenizer, T5ForConditionalGeneration

from training.spider_dataset import SpiderDataset
from preprocess.schema_utils import load_tables
from models.picard_interface import PicardDecoder


def evaluate_with_picard():
    # === Configuration ===
    model_dir = "output/t5-spider-final"
    dev_tsv = "output/processed/dev.tsv"
    tables_path = "data/spider_data/tables.json"
    db_path = "data/spider_data/database"
    save_output_to = "output/predictions/picard_dev_predictions.tsv"
    max_tokens = 100

    # === Setup ===
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load schema
    schemas = load_tables(tables_path)

    # Load dev set
    dataset = SpiderDataset(file_path=dev_tsv, tokenizer=tokenizer)

    # Setup Picard decoder
    picard = PicardDecoder(
        tokenizer=tokenizer,
        schemas=schemas,
        db_path=db_path,
        fix_issue_16_primary_keys=True  # Enable known fix if needed
    )

    predictions = []

    print(f"Evaluating on {len(dataset)} dev examples...")

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        db_id = dataset.examples[idx][2]  # Get DB ID from dataset
        input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

        # === Picard decoding ===
        with torch.no_grad():
            outputs = picard.decode(
                model=model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                db_id=db_id,
                max_length=max_tokens,
                num_beams=4,
                early_stopping=True
            )

        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ground_truth_sql = dataset.examples[idx][1]

        predictions.append((predicted_sql, ground_truth_sql))

        # Optional: print a few samples
        if idx < 5:
            print("\n" + "=" * 60)
            print("DB ID :", db_id)
            print("INPUT :", tokenizer.decode(sample["input_ids"], skip_special_tokens=True))
            print("PRED  :", predicted_sql)
            print("GT    :", ground_truth_sql)

    # Save predictions
    os.makedirs(os.path.dirname(save_output_to), exist_ok=True)
    with open(save_output_to, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["predicted_sql", "ground_truth_sql"])
        writer.writerows(predictions)

    print(f"\n✅ Saved {len(predictions)} predictions to {save_output_to}")


if __name__ == "__main__":
    evaluate_with_picard()
