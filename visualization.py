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
from visual_utils import get_attention_maps, plot_and_save_attention_map, sample_and_predict
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Attention Map Visualization')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to visualize attention maps')
    parser.add_argument('--max_length', type=int, default=64, help='Maximum length of input and attention map')
    parser.add_argument('--save_path', type=str, default='/home/juhwan/sangmin/bcell_active/attention_maps', help='Path to save attention maps')
    args = parser.parse_args()

    # random seed
    seed_everything(CFG['SEED'])

    # data load
    path = '/home/juhwan/sangmin/bcell_active/protein_data'
    all_df = pd.read_csv(path + '/train.csv')
    train_len = int(len(all_df) * 0.8)
    train_df = all_df.iloc[:train_len].reset_index(drop=True)
    val_df = all_df.iloc[train_len:].reset_index(drop=True)
    
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')
    num_samples = args.num_samples
    max_length = args.max_length

    # dataset & dataloader
    val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list = get_preprocessing('val', val_df)

    # Filter out sequences longer than max_length
    filtered_samples = [(e, l, r, label) for e, l, r, label in zip(val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list) if len(e + l + r) <= max_length]
    val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list = zip(*filtered_samples)

    val_dataset = CustomDataset(val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list, tokenizer, max_length=max_length)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)

    # model load
    model = BaseModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.load_state_dict(torch.load('./best_model.pth', map_location=device))
    model.eval()

    # visualization setting
    num_samples = 10  # number of samples to visualize attention maps
    save_base_path = args.save_path
    os.makedirs(save_base_path, exist_ok=True)

    # 여러 샘플에 대해 어텐션 맵 시각화
    save_base_path = 'attention_maps'
    os.makedirs(save_base_path, exist_ok=True)

    for i, sample in enumerate(val_dataset):
        if i >= num_samples:
            break

        sample, true_label, predicted_label, epitope_tokens, left_antigen_tokens, right_antigen_tokens = sample_and_predict(sample, model, device)

        # Attention Map 추출 및 시각화 저장
        tokens = {
            'input_ids': torch.cat((left_antigen_tokens['input_ids'], epitope_tokens['input_ids'], right_antigen_tokens['input_ids']), dim=1)
        }
        attention_maps = get_attention_maps(model, tokens)

        save_path = os.path.join(save_base_path, f'attention_map_sample_{i+1}')
        plot_and_save_attention_map(attention_maps, epitope_tokens, left_antigen_tokens, right_antigen_tokens, tokenizer, save_path, true_label, predicted_label)

        # 결과 출력
        print(f"Sample {i+1} - True Label: {true_label}, Predicted Label: {predicted_label}")
        print(f"Attention map saved to {save_path}")

# python visualization.py --num_samples 10 --max_length 64
if __name__ == '__main__':
    main()