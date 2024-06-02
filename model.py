from transformers import AutoModel
import torch
import torch.nn as nn
from config import CFG

class BaseModel(nn.Module):
    def __init__(self, freeze_esm=True, concat_embedding=False):
        super(BaseModel, self).__init__()
        
        self.esm_model = AutoModel.from_pretrained('facebook/esm1b_t33_650M_UR50S')
        
        if freeze_esm:
            for param in self.esm_model.parameters():
                param.requires_grad = False
        
        self.concat_embedding = concat_embedding
        in_channels = self.esm_model.config.hidden_size
        if not concat_embedding:
            in_channels *= 3

        self.classifier = nn.Sequential(
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels),
            nn.Linear(in_channels, in_channels // 4),
            nn.LeakyReLU(True),
            nn.BatchNorm1d(in_channels // 4),
            nn.Linear(in_channels // 4, 1)
        )

    def forward(self, epitope_x, left_antigen_x, right_antigen_x):
        if self.concat_embedding:
            combined_tokens = {
                'input_ids': torch.cat((epitope_x['input_ids'], left_antigen_x['input_ids'], right_antigen_x['input_ids']), dim=1),
                'attention_mask': torch.cat((epitope_x['attention_mask'], left_antigen_x['attention_mask'], right_antigen_x['attention_mask']), dim=1)
            }
            x = self.get_esm_embedding(combined_tokens)
        else:
            epitope_embedding = self.get_esm_embedding(epitope_x)
            left_antigen_embedding = self.get_esm_embedding(left_antigen_x)
            right_antigen_embedding = self.get_esm_embedding(right_antigen_x)
            x = torch.cat((epitope_embedding, left_antigen_embedding, right_antigen_embedding), dim=1)

        return self.classifier(x).view(-1)

    def get_esm_embedding(self, tokens):
        output = self.esm_model(**tokens)
        return output.last_hidden_state[:, 0, :]
