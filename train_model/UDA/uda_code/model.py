from transformers import RobertaModel
import torch
from torch import nn

class RobertaClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features):
        x = features[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class Calssifier(nn.Module):
    def __init__(self, model_path, config):
        super(Calssifier, self).__init__()
        self.config = config
        self.model = RobertaModel.from_pretrained(model_path,config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask, use_acc=False):
        if use_acc:
            bs,seq_len = input_ids.shape
            position_ids = torch.arange(0, seq_len).expand((1,-1)).to(input_ids.device)
            token_type_ids = torch.zeros(input_ids.shape, dtype=torch.long, device=input_ids.device)
            output = self.model(input_ids,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                                token_type_ids=token_type_ids,
                                return_dict=True)
        else:
            output = self.model(input_ids,
                    attention_mask=attention_mask,
                    return_dict=True)
        sequence_output = output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
