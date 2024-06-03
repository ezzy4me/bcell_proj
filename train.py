import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import f1_score
from config import CFG

def train(model, optimizer, train_loader, val_loader, scheduler, device):
    model.to(device)
    criterion = nn.BCEWithLogitsLoss().to(device)
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    best_val_f1 = 0

    for epoch in range(1, CFG['EPOCHS'] + 1):
        model.train()
        train_loss = 0
        train_preds = []
        train_labels = []

        for batch in tqdm(train_loader):
            epitope_seq = {key: val.to(device) for key, val in batch['epitope_tokens'].items()}
            left_antigen_seq = {key: val.to(device) for key, val in batch['left_antigen_tokens'].items()}
            right_antigen_seq = {key: val.to(device) for key, val in batch['right_antigen_tokens'].items()}
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                preds = model(epitope_seq, left_antigen_seq, right_antigen_seq)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            train_preds.extend(preds.sigmoid().cpu().detach().numpy())
            train_labels.extend(labels.cpu().detach().numpy())

        threshold = CFG['THRESHOLD']
        train_preds_binary = [1 if pred > threshold else 0 for pred in train_preds]
        train_f1 = f1_score(train_labels, train_preds_binary, average='macro')
        val_loss, val_f1 = validation(model, val_loader, criterion, device)

        print(f'Epoch {epoch}/{CFG["EPOCHS"]}, Train Loss: {train_loss / len(train_loader):.4f}, '
              f'Train F1: {train_f1:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}')

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(model.state_dict(), './best_model_1.pth', _use_new_zipfile_serialization=False)
            print('Model Saved.')
        
        if scheduler is not None:
            scheduler.step()

    return best_val_f1

def validation(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    val_preds = []
    val_labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader):
            epitope_seq = {key: val.to(device) for key, val in batch['epitope_tokens'].items()}
            left_antigen_seq = {key: val.to(device) for key, val in batch['left_antigen_tokens'].items()}
            right_antigen_seq = {key: val.to(device) for key, val in batch['right_antigen_tokens'].items()}
            labels = batch['labels'].to(device)

            with torch.cuda.amp.autocast(enabled=True):
                preds = model(epitope_seq, left_antigen_seq, right_antigen_seq)
                loss = criterion(preds, labels)

            val_loss += loss.item()
            val_preds.extend(preds.sigmoid().cpu().detach().numpy())
            val_labels.extend(labels.cpu().detach().numpy())

    threshold = CFG['THRESHOLD']
    val_preds_binary = [1 if pred > threshold else 0 for pred in val_preds]
    val_f1 = f1_score(val_labels, val_preds_binary, average='macro')
    return val_loss / len(val_loader), val_f1
