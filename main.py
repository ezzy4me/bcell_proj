import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
import pandas as pd
from config import seed_everything, CFG
from preprocessing import get_preprocessing
from dataset import CustomDataset, collate_fn
from model import BaseModel
from train import train
from inference import inference

def main():
    parser = argparse.ArgumentParser(description='B-cell Active Learning Model')
    parser.add_argument('--mode', type=str, choices=['train', 'inference'], required=True, help='Mode: train or inference')
    args = parser.parse_args()
    
    seed_everything(CFG['SEED'])
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    path = '/home/juhwan/sangmin/bcell_active/protein_data'
    
    if args.mode == 'train':
        all_df = pd.read_csv(path + '/train.csv')
        train_len = int(len(all_df) * 0.8)
        train_df = all_df.iloc[:train_len].reset_index(drop=True)
        val_df = all_df.iloc[train_len:].reset_index(drop=True)
        # train_df = all_df.iloc[:500].reset_index(drop=True)
        # val_df = all_df.iloc[500:600].reset_index(drop=True)

        tokenizer = AutoTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        max_length = CFG['ANTIGEN_MAX_LEN'] * 2 + CFG['EPITOPE_MAX_LEN']

        train_epitope_list, train_left_antigen_list, train_right_antigen_list, train_label_list = get_preprocessing('train', train_df)
        val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list = get_preprocessing('val', val_df)

        train_dataset = CustomDataset(train_epitope_list, train_left_antigen_list, train_right_antigen_list, train_label_list, tokenizer, max_length=max_length)
        train_loader = DataLoader(train_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=True, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)

        val_dataset = CustomDataset(val_epitope_list, val_left_antigen_list, val_right_antigen_list, val_label_list, tokenizer, max_length=max_length)
        val_loader = DataLoader(val_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)

        model = BaseModel()
        model.to(device)
        optimizer = torch.optim.Adam(params=model.parameters(), lr=CFG["LEARNING_RATE"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader) * CFG['EPOCHS'], eta_min=0)

        best_score = train(model, optimizer, train_loader, val_loader, scheduler, device)
        print(f'Best Validation F1 Score : [{best_score:.5f}]')

    elif args.mode == 'inference':
        test_df = pd.read_csv(path + '/test.csv')
        tokenizer = AutoTokenizer.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        max_length = CFG['ANTIGEN_MAX_LEN'] * 2 + CFG['EPITOPE_MAX_LEN']

        test_epitope_list, test_left_antigen_list, test_right_antigen_list, _ = get_preprocessing('test', test_df)
        
        test_dataset = CustomDataset(test_epitope_list, test_left_antigen_list, test_right_antigen_list, None, tokenizer, max_length=max_length)
        test_loader = DataLoader(test_dataset, batch_size=CFG['BATCH_SIZE'], shuffle=False, num_workers=CFG['NUM_WORKERS'], collate_fn=collate_fn)

        model = BaseModel()
        model.to(device)
        model.load_state_dict(torch.load('./best_model.pth', map_location=device))

        preds = inference(model, test_loader, device)
        
        submission = pd.read_csv(path + '/sample_submission.csv')
        submission['label'] = preds
        submission.to_csv('submission.csv', index=False)
        print('Inference and submission file saved.')

if __name__ == '__main__':
    main()
