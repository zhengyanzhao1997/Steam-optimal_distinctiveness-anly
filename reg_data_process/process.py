import nltk
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import json
import os
import numpy as np
import six
import random
import re
from cleantext import clean,to_ascii_unicode
from tqdm import tqdm
from transformers import RobertaTokenizer,RobertaConfig,RobertaModel
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from gensim.models.doc2vec import Doc2Vec
import re


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

os.environ["CUDA_VISIBLE_DEVICES"] = '1'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))


stoplist = stopwords.words('english')
punctuation = string.punctuation.replace('-','')


def clean_w2v_text(text):
    text = str(text)
    text = remove_urls(text)
    text = text.lower()  # Lowercase words
    text = re.sub(f"[{re.escape(punctuation)}]", " ", text) 
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    tokens = word_tokenize(text)  # Get tokens from text
    tokens = [t for t in tokens if not t in stoplist]  # Remove stopwords
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits
    tokens = [t.replace('-','') if t.startswith('-') else t for t in tokens]
    tokens = [t for t in tokens if len(t) > 1 and (t[0].isalpha() or t[0].isdigit())]  # Remove short tokens
    return tokens

for root, dirs, files in os.walk('./data'):
    review_app_list = [int(file.split('_')[1].split('.')[0]) for file in files]
    
game_infos = {}
with open('../game_info/cat_app_info_new.json','r') as f:
    for game in f:
        game_info = json.loads(game)
        game_infos[game_info['game_id']] = game_info
        

def clean_text(text):
    text = clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=False,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=True,                  # replace all URLs with a special token
        no_emails=True,                # replace all email addresses with a special token
        no_phone_numbers=True,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=False,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="<URL>",
        replace_with_email="<Email>",
        replace_with_phone_number="<Phone>",
        replace_with_number="<NUM>",
        replace_with_digit="0",
        replace_with_currency_symbol="<Symbol>",
        lang="en"
    )
    return text


def clean_text_f(text):
    text = clean(text,
        fix_unicode=True,               # fix various unicode errors
        to_ascii=True,                  # transliterate to closest ASCII representation
        lower=True,                     # lowercase text
        no_line_breaks=True,           # fully strip line breaks as opposed to only normalizing them
        no_urls=False,                  # replace all URLs with a special token
        no_emails=False,                # replace all email addresses with a special token
        no_phone_numbers=False,         # replace all phone numbers with a special token
        no_numbers=False,               # replace all numbers with a special token
        no_digits=False,                # replace all digits with a special token
        no_currency_symbols=False,      # replace all currency symbols with a special token
        no_punct=True,                 # remove punctuations
        replace_with_punct="",          # instead of removing punctuations you may replace them
        replace_with_url="",
        replace_with_email="",
        replace_with_phone_number="",
        replace_with_number="",
        replace_with_digit="",
        replace_with_currency_symbol="",
        lang="en"
    )
    return text


def filter_text(text):
    text = text.replace('—','-')
    f_text = clean_text_f(text)
    if len(f_text) > len(text)/2 or len(to_ascii_unicode(text)) == len(text):
        return clean_text(text)
    else:
        return None


def delete_link(text):
    len_text = len(text)
    relace_list = []
    i = 0
    while i < len_text-1:
        count = 0
        for j in range(i,len_text):
            if text[j] == text[i]:
                count += 1
            else:
                if count >= 8:
                    relace_list = [(i,j)] + relace_list
                i = j
                break
            if j == len_text - 1:
                if count >= 8:
                    relace_list = [(i,j+1)] + relace_list
                i = j
                break
    for i,j in relace_list:
        if i == 0 and j == len(text):
            return None
        if text[i] in '.?,~!' or text[i].isalpha():
            text = text[:i] + text[i]*random.randint(5,10) + text[j:]
        else:
            text = text[:i] + text[j:]
    if len(text) >= 3:
        return text
    else:
        return None
    

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
    def __init__(self, model_path, config, load_pre=True):
        super(PretrainModel, self).__init__()
        self.config = config
        if load_pre:
            self.model = RobertaModel.from_pretrained(model_path,config=config)
        else:
            self.model = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = RobertaClassificationHead(config)

    def forward(self, input_ids, attention_mask):
        output = self.model(input_ids, attention_mask=attention_mask,return_dict=True)
        sequence_output = output[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        return logits
    

# model_path_ori = './SENTIMENT'
# tokenizer_ori = RobertaTokenizer.from_pretrained(model_path_ori)
# config_ori = RobertaConfig.from_pretrained(model_path_ori)
# config_ori.num_labels = 5
# model_ori = PretrainModel(None, config_ori, False).to(device)
# model_ori.load_state_dict(torch.load('./SENTIMENT/best_model_10.pth'))
# model_ori.eval()


model_path_uda = 'siebert/sentiment-roberta-large-english'
tokenizer_uda = RobertaTokenizer.from_pretrained(model_path_uda)
config_uda = RobertaConfig.from_pretrained(model_path_uda)
config_uda.num_labels = 5
model_uda = PretrainModel(None, config_uda, False).to(device)
model_uda.load_state_dict(torch.load('./UDA/model/model_new.pth'))
model_uda.eval()


def get_token(comment,tokenizer):
    input = tokenizer(comment,max_length=512,truncation=True,return_tensors='pt',padding=True)
    return input


# @torch.no_grad()
# def get_sentiment_score(text,model1,model2,tokenizer1,tokenizer2):
#     text = delete_link(text)
#     if text:
#         ##### model_ori
#         input1 = get_token(text,tokenizer1)
#         output1 = model1(input1['input_ids'].to(device),input1['attention_mask'].to(device))
#         output1 = output1.argmax(-1)[0]
        
#         ##### model_uda
#         input2 = get_token(text,tokenizer2)
#         output2 = model2(input2['input_ids'].to(device),input2['attention_mask'].to(device))
#         output2 = output2.argmax(-1)[0]
        
#         return text,int(output1),int(output2)
#     else:
#         return None,None,None
    
# @torch.no_grad()
# def get_sentiment_score(text,model,tokenizer):
#     text = str(text)
#     text = delete_link(text)
#     if text:
#         ##### model_uda
#         input = get_token(text,tokenizer)
#         output = model(input['input_ids'].to(device),input['attention_mask'].to(device))
#         output = output.argmax(-1)[0]
#         return text,int(output)
#     else:
#         return None,None


def bindbatch(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]

@torch.no_grad()
def get_sentiment_score(texts,model,tokenizer):
    outputs = []
    for batch_text in bindbatch(texts,64):
        inputs = get_token(batch_text,tokenizer)
        output = model(inputs['input_ids'].to(device),inputs['attention_mask'].to(device))
        output = output.argmax(-1).tolist()
        outputs.extend(output)
    assert len(outputs) == len(texts)
    return [int(x) for x in outputs]
    
# def fetch_game_reviews(game_id,max_num,sentiment_count):
#     with open('./data/review_' + str(game_id) + '.json','r') as f:
#         c_data = json.load(f)
#         c_data = [v for k,v in c_data['reviews'].items() if v['author']['playtime_forever'] > 0]
#         c_data = sorted(c_data,key=lambda x:(-x['votes_up'],x['timestamp_created']))
#         output = []
#         count = 0
#         get_enough_sentiment = False
#         for d_ in c_data:
#             w2v_token = clean_w2v_text(d_['review'])
#             if count <= sentiment_count:
#                 sentiment_text,sentiment_score = get_sentiment_score(d_['review'],model_uda,tokenizer_uda)
#             else:
#                 sentiment_text,sentiment_score = None,None
#                 get_enough_sentiment = True
#             if (sentiment_text or get_enough_sentiment) and len(w2v_token) >= 1:
#                 count += 1
#                 d_['w2v_token'] = w2v_token
#                 d_['sentiment_text'] = sentiment_text
#                 d_['sentiment_score'] = sentiment_score
#                 output.append(d_)
#             if count >= max_num:
#                 break
#         return output

    
def fetch_game_reviews(game_id,max_num):
    with open('./data/review_' + str(game_id) + '.json','r') as f:
        c_data = json.load(f)
        
        vaild_data = []
        unvaild_data = []
        for d in c_data['reviews'].values():
            d['weighted_vote_score'] = float(d['weighted_vote_score'])
            d['votes_up'] = int(d['votes_up'])
            if d['weighted_vote_score'] > 0 or d['votes_up'] > 0:
                vaild_data.append(d)
            else:
                if 'playtime_at_review' not in d['author'].keys():
                    d['author']['playtime_at_review'] = 0
                else:
                    d['author']['playtime_at_review'] = int(d['author']['playtime_at_review'])
                unvaild_data.append(d)
        vaild_data = sorted(vaild_data,key=lambda x:(x['weighted_vote_score'],x['votes_up']),reverse=True)
        unvaild_data = sorted(unvaild_data,key=lambda x:x['author']['playtime_at_review'],reverse=True)
        c_data = vaild_data + unvaild_data

        output = []
        count = 0
        for d_ in c_data:
            w2v_token = clean_w2v_text(d_['review'])
            sentiment_text = delete_link(str(d_['review']))
            if sentiment_text and len(w2v_token) >= 1:
                count += 1
                d_['w2v_token'] = w2v_token
                d_['sentiment_text'] = sentiment_text
                output.append(d_)
            if count >= max_num:
                break
        sentiment_texts = [x['sentiment_text'] for x in output]
        sentiments_scores = get_sentiment_score(sentiment_texts,model_uda,tokenizer_uda)
        return output,sentiments_scores
    

doc_model = Doc2Vec.load('./cat_info/doc2vec_dm0_5_new/doc2vec_model_dm0_add_5_new')
words2index = doc_model.wv.key_to_index
vaild_word_set = list(words2index.keys())


def compute_vaild_token_num(tokens):
    return len(set([x for x in tokens if x in vaild_word_set]))
    
    
def gen_game_info(app_id,max_num):
    build_corpus = []
    g_info = game_infos[app_id]
    

    game_release_date = g_info['release_date']
    if not game_release_date or 'coming soon' in game_release_date:
        return None
    else:
        year = int(game_release_date.split('/')[0])
        if year < 2010:
            return None


    user_tag = g_info['usertag']
    game_tag = g_info['genre']
    dv_user_tag = ['user_comment_' + x for x in user_tag]
    dv_game_tag = ['game_desc_' + x for x in user_tag]
    app_description_short_token = clean_w2v_text(g_info['description_short'])
    app_description_long_token = clean_w2v_text(g_info['description_long'])
    long_vaild_num = compute_vaild_token_num(app_description_long_token)
    short_vaild_num = compute_vaild_token_num(app_description_short_token)
    
    if long_vaild_num < 10:
        print('unvaild app',app_id, short_vaild_num, long_vaild_num)
        return None
    
    if app_id in review_app_list:
        app_reviews,sentiments_scores = fetch_game_reviews(app_id,max_num)
    else:
        app_reviews,sentiments_scores = [],[]

    g_info['reviews'] = app_reviews
    g_info['sentiments_scores'] = sentiments_scores
    g_info['dv_user_tag'] = dv_user_tag
    g_info['dv_game_tag'] = dv_game_tag
    g_info['app_description_short_token'] = app_description_short_token
    g_info['app_description_long_token'] = app_description_long_token
    return g_info


game2cat = {}
with open('../game_info/cat_dict_vaild.json','r') as f:
    cat_total = []
    data = json.load(f)
    for cat_name,app_lists in data.items():
        for int_app_id in app_lists:
            if int_app_id in game2cat.keys():
                game2cat[int_app_id].append(cat_name)
            else:
                game2cat[int_app_id] = [cat_name]
        cat_total.extend(app_lists)
    cat_total = [x for x in list(set(cat_total))]
       

cat_total = []
with open('./vaild_app_lists.txt','r') as f:
    for i in f:
        cat_total.append(int(i.replace('\n','')))


with open('./total_detail.json','w') as f1:
    for app_id in tqdm(cat_total):
        g_info = gen_game_info(app_id,1024)
        if g_info:
            item = {'subcat':game2cat[app_id],'game_info':g_info}
            f1.write(json.dumps(item) + '\n')