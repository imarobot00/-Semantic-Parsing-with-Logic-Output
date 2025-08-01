from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from spider_dataset import SpiderDataset

def main():
    #  1. Load tokenizer and model
    model_name = "t5-small"  # or "t5-base" / "t5-large"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # 2. Load training data
    train_dataset = SpiderDataset(
        file_path="output/processed/train.tsv",
        tokenizer=tokenizer,
        max_input_length=512,
        max_target_length=256
    )

    training_args = TrainingArguments(
        output_dir="output/t5-spider-checkpoints",
        per_device_train_batch_size=4,
        num_train_epochs=2,
        logging_dir="output/logs",
        logging_steps=10
        # remove: save_strategy, evaluation_strategy, report_to
    )


    # 4. Setup HuggingFace Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer
    )

    #  5. Train the model
    trainer.train()
    model.save_pretrained("output/t5-spider-final")
    tokenizer.save_pretrained("output/t5-spider-final")
    print(" Training complete!")

if __name__ == "__main__":
    main()
