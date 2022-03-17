import json
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaTokenizer
import random

class SupDataset(Dataset):
    def __init__(self, sup_data_dir,tokenizer,input_maxlen):
        self.data = []
        with open(sup_data_dir) as f:
            for d in f:
                self.data.append(json.loads(d))
        self.tokenizer = tokenizer
        self.input_maxlen = input_maxlen
        self.label_convert = {0:0,1:1,2:1,3:1,4:2,5:2,6:2,7:3,8:3,9:3,10:4}

    def get_token(self,comment):
        input = self.tokenizer(comment,max_length=self.input_maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return input['input_ids'][0],input['attention_mask'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        grade = self.data[idx]['review_grade']
        label = self.label_convert[grade]
        comment_text = self.data[idx]['review_text']
        input_ids,attention_mask = self.get_token(comment_text)
        return (input_ids,attention_mask,label)


class UnsupDataset(Dataset):
    def __init__(self, unsup_data_dir,tokenizer,input_maxlen):
        self.data = []
        with open(unsup_data_dir) as f:
            for d in f:
                self.data.append(json.loads(d))
        self.tokenizer = tokenizer
        self.input_maxlen = input_maxlen

    def get_token(self,comment):
        input = self.tokenizer(comment,max_length=self.input_maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return input['input_ids'][0],input['attention_mask'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ori_text = self.data[idx]['clean_text']
        aug_text = self.data[idx]['augmented_text']
        if isinstance(aug_text,list):
            aug_text = random.choice(aug_text)    
        un_label = int(self.data[idx]['voted_up'])
        ori_input_ids,ori_attention_mask = self.get_token(ori_text)
        aug_input_ids, aug_attention_mask = self.get_token(aug_text)
        return (ori_input_ids,ori_attention_mask,aug_input_ids, aug_attention_mask,un_label)


class EvalDataset(Dataset):
    def __init__(self, eval_data_dir,tokenizer,input_maxlen):
        self.data = []
        with open(eval_data_dir) as f:
            for d in f:
                self.data.append(json.loads(d))
        self.tokenizer = tokenizer
        self.input_maxlen = input_maxlen

    def get_token(self,comment):
        input = self.tokenizer(comment,max_length=self.input_maxlen,truncation=True,padding='max_length',return_tensors='pt')
        return input['input_ids'][0],input['attention_mask'][0]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]['clean_text']
        label = int(self.data[idx]['voted_up'])
        input_ids,attention_mask = self.get_token(text)
        return (input_ids,attention_mask,label)


class load_data:
    def __init__(self, cfg):
        self.cfg = cfg
        self.tokenizer = RobertaTokenizer.from_pretrained(cfg['model_path'])
        self.input_maxlen = cfg['max_seq_length']
        if cfg['mode'] == 'train':
            self.sup_data_dir = cfg['sup_data_dir']
            self.sup_batch_size = cfg['train_batch_size']
            self.shuffle = True
        elif cfg['mode'] == 'train_eval':
            self.sup_data_dir = cfg['sup_data_dir']
            self.eval_data_dir = cfg['eval_data_dir']
            self.sup_batch_size = cfg['train_batch_size']
            self.eval_batch_size = cfg['eval_batch_size']
            self.shuffle = True
        elif cfg['mode'] == 'eval':
            self.sup_data_dir = cfg['eval_data_dir']
            self.sup_batch_size = cfg['eval_batch_size']
            self.shuffle = False  # Not shuffel when eval mode

        self.unsup_data_dir = cfg['unsup_data_dir']
        self.unsup_batch_size = cfg['train_batch_size'] * cfg['unsup_ratio']

    def sup_data_iter(self):
        sup_dataset = SupDataset(self.sup_data_dir,self.tokenizer,self.input_maxlen)
        sup_data_iter = DataLoader(sup_dataset, batch_size=self.sup_batch_size, shuffle=self.shuffle)
        return sup_data_iter

    def unsup_data_iter(self):
        unsup_dataset = UnsupDataset(self.unsup_data_dir,self.tokenizer,self.input_maxlen)
        unsup_data_iter = DataLoader(unsup_dataset, batch_size=self.unsup_batch_size, shuffle=self.shuffle)
        return unsup_data_iter

    def eval_data_iter(self):
        eval_dataset = SupDataset(self.eval_data_dir,self.tokenizer,self.input_maxlen)
        eval_data_iter = DataLoader(eval_dataset, batch_size=self.eval_batch_size, shuffle=False)
        return eval_data_iter
