import nltk
import string
from nltk.corpus import stopwords
from nltk import word_tokenize
import re
import json
import os
from tqdm import tqdm
import numpy as np
import six
import re


def remove_urls(vTEXT):
    vTEXT = re.sub(r'(https|http)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '', vTEXT, flags=re.MULTILINE)
    return(vTEXT)

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
    app_list = [int(file.split('_')[1].split('.')[0]) for file in files]
    
    
# def get_game_reviews_tokens(game_id):
#     with open('./data/review_' + str(game_id) + '.json','r') as f:
#         c_data = json.load(f)
#         review = c_data['reviews']
#         play_time = [v['author']['playtime_forever'] for k,v in review.items()]
#         play_time = sorted(play_time)
#         if len(play_time) > 0:
#             if len(play_time) >= 10000:
#                 min_play_time = play_time[int(len(play_time)/20)]
#             else:
#                 min_play_time = 0
#             reviews = [clean_w2v_text(v['review']) for k,v in review.items() if v['author']['playtime_forever'] > min_play_time]
#             reviews = [x for x in reviews if len(x) >= 20]
#             return reviews
#         else:
#             return []
        
def get_game_reviews_tokens(game_id):
    with open('./data/review_' + str(game_id) + '.json','r') as f:
        c_data = json.load(f)
        review_num = len(list(c_data['reviews'].keys()))
        if review_num > 0:
            count = 0
            reviews = []
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
            for v in c_data:
                tokens = clean_w2v_text(v['review'])
                if len(set(tokens)) >= 10:
                    reviews.append(tokens)
                    count += 1
                    if count >= 50000:
                        break
            return reviews
        else:
            return []


count = 0

vaild_tags = []
with open('./vaild_tags.txt','r') as f:
    for i in f:
        tag = i.strip('\n')
        vaild_tags.append(str(tag))


with open('./corpus_add_8.json','w') as f1:
    with open('../game_info/cat_app_info_new.json','r') as f:
        for game in tqdm(f):
            build_corpus = []
            game_info = json.loads(game)
            if game_info['request_status'] == 'Available' and 'About This Game' in game_info['description_title']:
                user_tag = [x for x in game_info['usertag'] if x in vaild_tags][:5]
                game_tag = game_info['genre']
                app_description_short = clean_w2v_text(game_info['description_short'])
                if len(app_description_short) >=5:
                    build_corpus.append({'type':'game_desc','text':app_description_short,'tag':game_tag})

                app_description_long = clean_w2v_text(game_info['description_long'])
                if len(app_description_long) >=10:
                    build_corpus.append({'type':'game_desc','text':app_description_long,'tag':game_tag})            

                app_id = int(game_info['game_id'])
                if app_id in app_list:
                    app_reviews = get_game_reviews_tokens(app_id)
                    get_reviews_num = len(app_reviews)
                    if get_reviews_num > 0:
                        build_corpus.extend([{'type':'user_comment','text':review,'tag':user_tag} for review in app_reviews])
                count += len(build_corpus)
                output = {'app_id':app_id,'corpus':build_corpus}
                f1.write(json.dumps(output) + '\n') 

print('total corpus num',count)