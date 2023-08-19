from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
import pandas as pd
from transformers import T5Tokenizer
import torch

class TLDataset(Dataset):
    def __init__(self, dataframe, tokenizer, source_len=512):
        self.tokenizer = tokenizer
        self.data = dataframe.reset_index()
        self.source_len = source_len
        self.summ_len = 16
        self.text = self.data["input_text"]
        self.targets = self.data["labels"]
        
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        src = self.text[index]
        trg = str(self.targets[index])
        
        source = self.tokenizer.batch_encode_plus([src], padding="max_length", max_length = self.source_len, truncation = True,return_tensors='pt')
        target = self.tokenizer.batch_encode_plus([trg], padding="max_length", max_length = self.summ_len, truncation = True,return_tensors='pt')

        source_ids = source['input_ids'].squeeze()
        source_mask = source['attention_mask'].squeeze()
        target_ids = target['input_ids'].squeeze()
        target_mask = target['attention_mask'].squeeze()

        return {
            'source_ids': source_ids.to(dtype=torch.long), 
            'source_mask': source_mask.to(dtype=torch.long), 
            'target_ids': target_ids.to(dtype=torch.long),
            'target_mask': target_mask.to(dtype=torch.long)
        }