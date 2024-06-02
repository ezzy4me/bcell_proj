import numpy as np
import torch
from tqdm import tqdm
from config import CFG

def inference(model, test_loader, device):
    model.eval()
    pred_proba_label = []
    with torch.no_grad():
        for batch in tqdm(test_loader):
            epitope_seq = {key: val.to(device) for key, val in batch['epitope_tokens'].items()}
            left_antigen_seq = {key: val.to(device) for key, val in batch['left_antigen_tokens'].items()}
            right_antigen_seq = {key: val.to(device) for key, val in batch['right_antigen_tokens'].items()}
            
            model_pred = model(epitope_seq, left_antigen_seq, right_antigen_seq)
            model_pred = torch.sigmoid(model_pred).to('cpu')
            
            pred_proba_label += model_pred.tolist()
    
    pred_label = np.where(np.array(pred_proba_label) > CFG['THRESHOLD'], 1, 0)
    return pred_label
