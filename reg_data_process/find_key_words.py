from gensim.models.doc2vec import Doc2Vec
import json
from tqdm import tqdm
import numpy as np
from nltk import sent_tokenize
from nltk import word_tokenize
import re
import os

keyword_dict = {}
with open('./core_word.json','r') as f:
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
    tokens = list(set(word_tokenize(text)))
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
                        key_find.append((k,key_word,sent_text,sentences[idx-1][0] if idx >= 1 else "", \
                                         sentences[idx+1][0] if idx < sent_num - 1 else ""))
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


cat_total_num = {}

year_range = [2019,2021]

dir_ = './subcat_no_embed_' + str(year_range[0]) + '_' + str(year_range[1]) + '/'
if not os.path.exists(dir_):
    os.makedirs(dir_)

with open('./total_detail.json','r') as f1:
    for app in tqdm(f1):
        app = json.loads(app)
        subcat = app['subcat']
        game_info = app['game_info']
        
        game_release_date = game_info['release_date']
        if not game_release_date or 'coming soon' in game_release_date:
            continue
        else:
            year = int(game_release_date.split('/')[0])
            if year > year_range[1] or year < year_range[0]:
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
        for sub in subcat:
            if sub in cat_total_num.keys():
                cat_total_num[sub] += 1
            else:
                cat_total_num[sub] = 1
            with open(dir_ + sub + '.json','a+') as f:
                f.write(json.dumps(output_item) + '\n')

print('result',cat_total_num)