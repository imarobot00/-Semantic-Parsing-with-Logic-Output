import os
import csv
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from training.spider_dataset import SpiderDataset
from tqdm import tqdm


def evaluate():
    # === Configuration ===
    model_dir = "output/t5-spider-final"
    dev_tsv = "output/processed/dev.tsv"
    save_output_to = "output/predictions/t5_dev_predictions.tsv"
    max_tokens = 100

    # === Setup ===
    tokenizer = T5Tokenizer.from_pretrained(model_dir)
    model = T5ForConditionalGeneration.from_pretrained(model_dir)
    model.eval()
    model.to("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpiderDataset(file_path=dev_tsv, tokenizer=tokenizer)
    predictions = []

    print(f"Evaluating on {len(dataset)} dev examples...")

    for idx in tqdm(range(len(dataset))):
        sample = dataset[idx]
        input_ids = sample["input_ids"].unsqueeze(0).to(model.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=max_tokens,
                num_beams=4,
                early_stopping=True
            )

        predicted_sql = tokenizer.decode(outputs[0], skip_special_tokens=True)
        ground_truth_sql = dataset.examples[idx][1]

        predictions.append((predicted_sql, ground_truth_sql))

        # Optional: print first few predictions
        if idx < 5:
            print("\n" + "="*60)
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
    evaluate()
