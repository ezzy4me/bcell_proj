import random
import numpy as np
import torch
import os

CFG = {
    'NUM_WORKERS': 4,
    'ANTIGEN_WINDOW': 256,
    'ANTIGEN_MAX_LEN': 256,
    'EPITOPE_MAX_LEN': 32,
    'EPOCHS': 4,
    'LEARNING_RATE': 5e-5,
    'BATCH_SIZE': 192,
    'THRESHOLD': 0.5,
    'SEED': 41
}

def seed_everything(seed=CFG['SEED']):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
