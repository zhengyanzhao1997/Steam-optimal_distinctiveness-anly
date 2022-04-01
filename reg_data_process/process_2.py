from gensim.models.doc2vec import Doc2Vec
import json
from tqdm import tqdm
import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize
import re
import os
from transformers import pipeline
import torch


os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))


sentiment_analysis = pipeline("sentiment-analysis",model="fine_grained_sentiment_model",device=0)


def bindbatch(listTemp, n):
    for i in range(0, len(listTemp), n):
        yield listTemp[i:i + n]


@torch.no_grad()
def model_pred_sentiment(texts):
    results = []
    for batch in bindbatch(texts,32):
        result = [1 if x['label'] == 'POSITIVE' else 0 for x in sentiment_analysis(batch,max_length=512,truncation=True)]
        results.extend(result)
    assert len(results) == len(texts)
    return results


keyword_dict = {}
with open('core_word.json', 'r') as f:
    for i in f:
        item = json.loads(i)
        title = item['title']
        key_words = list(set(item['new_key_words']))
        keyword_dict[title] = key_words


import string
punctuation = string.punctuation.replace('-','')
def sentences_tokenize(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(punctuation)}]", " ", text)
    text = re.sub(r"\s+", " ", text)
    tokens = word_tokenize(text)
    return tokens


def clean_w2v_text(text):
    text = re.sub(r"\n+", ".", text)
    text = re.sub(r"\s+", " ", text)
    sentences = sent_tokenize(text)
    output = []
    for sent in sentences:
        tokens = sentences_tokenize(sent)
        output.append((sent,tokens))
    return output


def find_comment_key_words_sentenpice(text,token_lists):
    key_find = []
    sentences = clean_w2v_text(text)
    sent_num = len(sentences)
    for k,v in keyword_dict.items():
        for key_word in v:
            if key_word in token_lists:
                for idx, (sent_text,sent_token) in enumerate(sentences):
                    if key_word in sent_token:
                        key_find.append({'title':k,'key_word':key_word,'sentence':sent_text,'sentiment':None})
    sentiment_scores = model_pred_sentiment([x['sentence'] for x in key_find])
    assert len(sentiment_scores) == len(key_find)
    for score,d in zip(sentiment_scores,key_find):
        d['sentiment'] = score
    return key_find


def find_game_key_words(token_lists):
    key_find = []
    for k,v in keyword_dict.items():
        for key_word in v:
            if key_word in token_lists:
                key_find.append((k,key_word))
    return key_find


vaild_tags = []
with open('../vaild_tags.txt','r') as f:
    for i in f:
        tag = i.strip('\n')
        vaild_tags.append(str(tag))


exit_apps = []
with open('./final_key_words_220_3.json','r') as f:
    for i in f:
        exit_apps.append(json.loads(i)['game_info']['game_id'])

        
print(len(exit_apps))


with open('./total_detail.json','r') as f1:
    for app in tqdm(f1):
        app = json.loads(app)
        subcat = app['subcat']
        game_info = app['game_info']  
        
        game_id = game_info['game_id']
        if game_id in exit_apps:
            continue
        # get info
        app_long_tokens = game_info['app_description_long_token']
        app_short_tokens = game_info['app_description_short_token']
        game_info['dv_user_tag'] = ['user_comment_' + x for x in game_info['usertag'] if x in vaild_tags]
        reviews = game_info['reviews']
        game_key_word = find_game_key_words(app_short_tokens + app_long_tokens)
        game_info['game_key_word'] = game_key_word
        new_reviews = []
        for review in reviews:
            review_key_word = find_comment_key_words_sentenpice(review['review'],review['w2v_token'])
            review['key_words'] = review_key_word
            new_reviews.append(review)
        assert len(game_info['sentiments_scores']) == len(new_reviews)
        game_info['reviews'] = new_reviews
        output_item = {'subcat':subcat,'game_info':game_info}
        with open('./final_key_words.json','a+') as f2:
            f2.write(json.dumps(output_item) + '\n')