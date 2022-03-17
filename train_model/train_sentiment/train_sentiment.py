import random
from transformers import RobertaTokenizer,RobertaConfig,RobertaModel
import json
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

seed = 1024

def init_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)

init_seed(seed)
model_path = "./sitiment_roberta"
tokenizer = RobertaTokenizer.from_pretrained(model_path)
config = RobertaConfig.from_pretrained(model_path)
config.num_labels = 5
input_maxlen = 512
batch_size = 32
learning_rate = 2e-5
epochs = 10

train_data_path = './train_data.json'
test_data_path = './test_data.json'

def load_data(path):
    data = []
    with open(path) as f:
        for d in f:
            data.append(json.loads(d))
    return data


class GetDataset(Dataset):
    def __init__(self, data, tokenizer, input_maxlen, use_game=False, transform=None, target_transform=None):
        self.data = data
        self.tokenizer = tokenizer
        self.input_maxlen = input_maxlen
        self.label_convert = {0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4}
        self.use_game = use_game
        self.transform = transform
        self.target_transform = target_transform

    def get_token(self,comment,game_text):
        if self.use_game:
            input = self.tokenizer(comment,game_text,max_length=self.input_maxlen,truncation=True,padding='max_length',return_tensors='pt')
        else:
            input = self.tokenizer(comment,max_length=self.input_maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return input

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grade = self.data[idx]['review_grade']
        label = self.label_convert[grade]
        comment_text = self.data[idx]['review_text']
        game_text = self.data[idx]['game_info']['summary']
        input = self.get_token(comment_text,game_text)
        input['input_ids'] = input['input_ids'][0]
        input['attention_mask'] = input['attention_mask'][0]
        return input,label


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


class PretrainModel(nn.Module):
    def __init__(self, model_path, config):
        super(PretrainModel, self).__init__()
        self.config = config
        self.model = RobertaModel.from_pretrained(model_path,config=config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,return_dict=True)
        sequence_output = output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits


loss_fct = nn.CrossEntropyLoss()


def train(dataloader, model, optimizer, scheduler):
    model.train()
    for batch, (inputs,labels) in enumerate(tqdm(dataloader,desc='training')):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        loss = loss_fct(logits.view(-1,config.num_labels), labels.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        model.zero_grad()
        if batch % 100 == 0:
            print('step',batch,'loss',loss)

def compute_f1_ma(pred_result,y,tag):
    TP = ((pred_result == tag) & (y == tag)).type(torch.float).sum().item()
    TN = ((pred_result != tag) & (y != tag)).type(torch.float).sum().item()
    FN = ((pred_result != tag) & (y == tag)).type(torch.float).sum().item()
    FP = ((pred_result == tag) & (y != tag)).type(torch.float).sum().item()
    return TP,TN,FN,FP

@torch.no_grad()
def test(dataloader, model):
    model.eval()
    TP_0, TN_0, FN_0, FP_0 = 0, 0, 0, 0
    TP_1, TN_1, FN_1, FP_1 = 0, 0, 0, 0
    TP_2, TN_2, FN_2, FP_2 = 0, 0, 0, 0
    TP_3, TN_3, FN_3, FP_3 = 0, 0, 0, 0
    TP_4, TN_4, FN_4, FP_4 = 0, 0, 0, 0

    for batch, (inputs,labels) in enumerate(tqdm(dataloader,desc='testing')):
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        labels = labels.to(device)
        logits = model(input_ids, attention_mask)
        logits = logits.view(-1, config.num_labels)
        pred = logits.argmax(-1)
        label = labels.view(-1)

        TP,TN,FN,FP = compute_f1_ma(pred,label,0)
        TP_0 += TP
        TN_0 += TN
        FN_0 += FN
        FP_0 += FP

        TP,TN,FN,FP = compute_f1_ma(pred,label,1)
        TP_1 += TP
        TN_1 += TN
        FN_1 += FN
        FP_1 += FP

        TP,TN,FN,FP = compute_f1_ma(pred,label,2)
        TP_2 += TP
        TN_2 += TN
        FN_2 += FN
        FP_2 += FP

        TP,TN,FN,FP = compute_f1_ma(pred,label,3)
        TP_3 += TP
        TN_3 += TN
        FN_3 += FN
        FP_3 += FP

        TP,TN,FN,FP = compute_f1_ma(pred,label,4)
        TP_4 += TP
        TN_4 += TN
        FN_4 += FN
        FP_4 += FP

    P_0 = TP_0 / (TP_0 + FP_0)
    R_0 = TP_0 / (TP_0 + FN_0)

    P_1 = TP_1 / (TP_1 + FP_1)
    R_1 = TP_1 / (TP_1 + FN_1)

    P_2 = TP_2 / (TP_2 + FP_2)
    R_2 = TP_2 / (TP_2 + FN_2)

    P_3 = TP_3 / (TP_3 + FP_3)
    R_3 = TP_3 / (TP_3 + FN_3)

    P_4 = TP_4 / (TP_4 + FP_4)
    R_4 = TP_4 / (TP_4 + FN_4)

    F_0 = 2 * R_0 * P_0 / (R_0 + P_0)
    F_1 = 2 * R_1 * P_1 / (R_1 + P_1)
    F_2 = 2 * R_2 * P_2 / (R_2 + P_2)
    F_3 = 2 * R_3 * P_3 / (R_3 + P_3)
    F_4 = 2 * R_4 * P_4 / (R_4 + P_4)

    F = (F_0+F_1+F_2+F_3+F_4)/5
    print('F0', F_0)
    print('F1', F_1)
    print('F2', F_2)
    print('F3', F_3)
    print('F4', F_4)
    print('F',F)
    return F


if __name__ == '__main__':
    train_data = load_data(train_data_path)
    test_data = load_data(test_data_path)
    random.shuffle(train_data)
    print(len(train_data))
    print(len(test_data))
    model = PretrainModel(model_path, config)
    model = nn.DataParallel(model, device_ids=[0, 1, 2, 3]).to(device)
    training_data = GetDataset(train_data, tokenizer, input_maxlen)
    train_dataloader = DataLoader(training_data, batch_size=batch_size,shuffle=True,drop_last=True)
    testing_data = GetDataset(test_data, tokenizer, input_maxlen)
    test_dataloader = DataLoader(testing_data, batch_size=batch_size,drop_last=True)
    num_training_steps = epochs * len(training_data) // batch_size
    num_warmup_steps = int(num_training_steps * 0.1)
    print('train num_training_steps:', num_training_steps)
    print('num_warmup_steps', num_warmup_steps)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    F1max = 0
    for t in range(epochs):
        print(f"Train Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, optimizer, scheduler)
        F1 = test(test_dataloader, model)
        if F1 > F1max:
            F1max = F1
            torch.save(model.module.state_dict(), "./model_saved/best_model_10.pth")
            print(f"Higher F1: {(F1max):>5f}%, Saved PyTorch Model State to model.pth")
    print("Training done!")
