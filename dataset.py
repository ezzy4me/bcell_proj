import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
import torch

class CustomDataset(Dataset):
    def __init__(self, epitope_list, left_antigen_list, right_antigen_list, label_list, tokenizer, max_length=512):
        self.epitope_list = epitope_list
        self.left_antigen_list = left_antigen_list
        self.right_antigen_list = right_antigen_list
        self.label_list = label_list
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index):
        epitope = self.epitope_list[index]
        left_antigen = self.left_antigen_list[index]
        right_antigen = self.right_antigen_list[index]

        epitope_tokens = self.tokenizer(epitope, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        left_antigen_tokens = self.tokenizer(left_antigen, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)
        right_antigen_tokens = self.tokenizer(right_antigen, return_tensors='pt', padding='max_length', truncation=True, max_length=self.max_length)

        item = {
            'epitope_tokens': epitope_tokens,
            'left_antigen_tokens': left_antigen_tokens,
            'right_antigen_tokens': right_antigen_tokens
        }
        if self.label_list is not None:
            item['labels'] = torch.tensor(self.label_list[index], dtype=torch.float)
        
        return item

    def __len__(self):
        return len(self.epitope_list)

def collate_fn(batch):
    epitope_input_ids = torch.stack([item['epitope_tokens']['input_ids'].squeeze() for item in batch])
    epitope_attention_mask = torch.stack([item['epitope_tokens']['attention_mask'].squeeze() for item in batch])

    left_antigen_input_ids = torch.stack([item['left_antigen_tokens']['input_ids'].squeeze() for item in batch])
    left_antigen_attention_mask = torch.stack([item['left_antigen_tokens']['attention_mask'].squeeze() for item in batch])

    right_antigen_input_ids = torch.stack([item['right_antigen_tokens']['input_ids'].squeeze() for item in batch])
    right_antigen_attention_mask = torch.stack([item['right_antigen_tokens']['attention_mask'].squeeze() for item in batch])

    output = {
        'epitope_tokens': {
            'input_ids': epitope_input_ids,
            'attention_mask': epitope_attention_mask
        },
        'left_antigen_tokens': {
            'input_ids': left_antigen_input_ids,
            'attention_mask': left_antigen_attention_mask
        },
        'right_antigen_tokens': {
            'input_ids': right_antigen_input_ids,
            'attention_mask': right_antigen_attention_mask
        }
    }

    if 'labels' in batch[0]:
        output['labels'] = torch.stack([item['labels'] for item in batch])

    return output
