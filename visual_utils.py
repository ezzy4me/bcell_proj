import pandas as pd
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from config import seed_everything, CFG
from preprocessing import get_preprocessing
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset import CustomDataset, collate_fn
from model import BaseModel

# Attenion map extraction for visualization
def get_attention_maps(model, tokens):
    model.eval()
    with torch.no_grad():
        output = model.esm_model(**tokens, output_attentions=True)
        attention_maps = output.attentions  # 모든 레이어의 attention map을 반환
    return attention_maps

# random sample and predict
def sample_and_predict(sample, model, device):
    sample_loader = DataLoader([sample], batch_size=1, shuffle=True, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)
    sample = next(iter(sample_loader))
    
    epitope_seq = {key: val.to(device) for key, val in sample['epitope_tokens'].items()}
    left_antigen_seq = {key: val.to(device) for key, val in sample['left_antigen_tokens'].items()}
    right_antigen_seq = {key: val.to(device) for key, val in sample['right_antigen_tokens'].items()}
    true_label = sample['labels'].item()

    model.eval()
    with torch.no_grad():
        model_pred = model(epitope_seq, left_antigen_seq, right_antigen_seq)
        model_pred = torch.sigmoid(model_pred).to('cpu')
        predicted_label = 1 if model_pred > CFG['THRESHOLD'] else 0

    return sample, true_label, predicted_label, epitope_seq, left_antigen_seq, right_antigen_seq


def plot_and_save_attention_map(attention_maps, epitope_tokens, left_antigen_tokens, right_antigen_tokens, tokenizer, save_path, true_label, predicted_label, num_heads=12):
    # combine tokens
    combined_tokens = torch.cat((left_antigen_tokens['input_ids'], epitope_tokens['input_ids'], right_antigen_tokens['input_ids']), dim=1) # (1, seq_len)
    combined_tokens = combined_tokens.squeeze().cpu().numpy()
    token_strs = tokenizer.convert_ids_to_tokens(combined_tokens.flatten())

    # range of epitope tokens
    epitope_start = len(left_antigen_tokens['input_ids'][0])
    epitope_end = epitope_start + len(epitope_tokens['input_ids'][0])
    epitope_range = range(epitope_start, epitope_end)

    # remove padding tokens
    pad_token_id = tokenizer.pad_token_id
    valid_indices = np.where(combined_tokens != pad_token_id)[0]

    # attention map of the last layer
    attention_map = attention_maps[-1][0].cpu().numpy()
    
    # attention map for valid tokens without padding tokens
    attention_map = attention_map[:, valid_indices, :][:, :, valid_indices]
    token_strs = [token_strs[i] for i in valid_indices]

    # visualize attention map for each head
    for head in range(num_heads):
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(attention_map[head], xticklabels=token_strs, yticklabels=token_strs, ax=ax, cmap="coolwarm", vmin=0, vmax=1)
        
        # Epitope tokens에 대해서만 색 변경
        for i in epitope_range:
            if i in valid_indices:  # 유효 인덱스에 있는 경우에만 색 변경
                adjusted_index = np.where(valid_indices == i)[0][0]
                ax.get_xticklabels()[adjusted_index].set_color("red")
                ax.get_yticklabels()[adjusted_index].set_color("red")
        
        plt.title(f"Attention Map - Last Layer - Head {head+1}\nTrue Label: {true_label}, Predicted Label: {predicted_label}")
        plt.savefig(f"{save_path}_head_{head+1}.png")
        plt.close()