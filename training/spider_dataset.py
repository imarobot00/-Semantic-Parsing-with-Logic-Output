#to load the tsv file and tokenize the input and target pairs and returns batches ready for training Hugging Face models

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer
import csv

class SpiderDataset(Dataset):
    def __init__(self, file_path,tokenizer: PreTrainedTokenizer, max_input_length=512, max_target_length=256):
        self.examples = []
        self.tokenizer = tokenizer

        with open(file_path, 'r', encoding='utf-8') as f:
            reader=csv.DictReader(f, delimiter='\t')
            for row in reader:
                # Handle both old format (input, target) and new format (input, target, db_id)
                if 'db_id' in row:
                    self.examples.append((row['input'], row['target'], row['db_id']))
                else:
                    self.examples.append((row['input'], row['target'], 'unknown'))

        self.max_input_length = max_input_length
        self.max_target_length = max_target_length
    
    def __len__(self): #To tell the dataloader that the dataset has X training samples
        return len(self.examples) #Telss how manu batcheds to run per epoch
    
    def __getitem__(self, idx):
        # Handle both old format (input, target) and new format (input, target, db_id)
        if len(self.examples[idx]) >= 3:
            input_text, target_text, db_id = self.examples[idx][:3]
        else:
            input_text, target_text = self.examples[idx][:2]
            db_id = 'unknown'

        # Tokenize input and target
        input_enc = self.tokenizer(
            input_text,
            max_length=self.max_input_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        target_enc = self.tokenizer(
            target_text,
            max_length=self.max_target_length,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )

        return {
            'input_ids': input_enc['input_ids'].squeeze(0),
            'attention_mask': input_enc['attention_mask'].squeeze(0),
            'labels': target_enc['input_ids'].squeeze(0)
        }

