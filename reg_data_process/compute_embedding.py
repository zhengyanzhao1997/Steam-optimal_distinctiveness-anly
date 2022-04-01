import os
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from sentence_embedding import get_tfidf_embedding,get_mean_embedding
from transformers import pipeline
from tqdm import tqdm
import json
import time

# import torch
# os.environ["CUDA_VISIBLE_DEVICES"] = '0'
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print("Using {} device".format(device))


# sentiment_analysis = pipeline("sentiment-analysis",model="siebert/sentiment-roberta-large-english",device=0)


model_1 = Doc2Vec.load('./doc2vec_8/doc2vec_model_dm1_8')
model_2 = Doc2Vec.load('./doc2vec_8/doc2vec_model_dm0_8')


feature_dict =  {'story': 0,
                 'graphic': 1,
                 'video': 2,
                 'audio': 3,
                 'vibe': 4,
                 'world': 5,
                 'scenery': 6,
                 'model': 7,
                 'ui': 8,
                 'tutorial': 9,
                 'fight': 10,
                 'props': 11,
                 'puzzle': 12,
                 'boss': 13,
                 'achievement': 14,
                 'character': 15,
                 'operation': 16,
                 'engine': 17,
                 'missions': 18,
                 'mode': 19}

feature_nums = len(list(feature_dict.keys()))

def get_cos_similar(v1, v2):
    num = float(np.dot(v1, v2))  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2)  # 求模长的乘积
    return 0.5 + 0.5 * (num / denom) if denom != 0 else 0


def get_cos_similar_multi(v1, v2):
    num = np.dot([v1], np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def get_cos_similar_matrix(v1, v2):
    num = np.dot(v1, np.array(v2).T)  # 向量点乘
    denom = np.linalg.norm(v1, axis=1).reshape(-1, 1) * np.linalg.norm(v2, axis=1)  # 求模长的乘积
    res = num / denom
    res[np.isneginf(res)] = 0
    return 0.5 + 0.5 * res


def get_score(embeddings,app_time_lists,topp=0.05,min_compare=5):
    embeddings = np.array(embeddings)
    assert len(embeddings) == len(app_time_lists)
    app_num = len(embeddings)
    average_scores = []
    top_scores = []
    for idx,embed in enumerate(embeddings):
        pub_times = app_time_lists[idx]
        if pub_times == -1:
            top_scores.append(None)
            average_scores.append(None)
            continue
        before_index = np.where(np.array(app_time_lists) < pub_times)[0]
        vaild_top_num = int(len(before_index) * topp)
        if vaild_top_num >= min_compare:
            vaild_embeddings = embeddings[before_index]
            average_embeddings = np.mean(vaild_embeddings, axis=0)
            average_socre = get_cos_similar(embed, average_embeddings)
            average_scores.append(average_socre)
            top_embeddings = embeddings[before_index][:vaild_top_num]
            top_score = np.max(get_cos_similar_multi(embed,top_embeddings))
            top_scores.append(top_score)
        else:
            top_scores.append(None)
            average_scores.append(None)
    assert len(top_scores) == len(average_scores) == app_num
    return average_scores,top_scores
    

def get_spy_tag_score(embeddings,ori_tag_lists,spy_tag_lists,dv,way='preweight',vaild_num=None):
    tag_scores = []
    spy_vailds = []
    tags_labeled_nums = []
    app_num = len(embeddings)
    for embedding,ori_tags,spy_tags in zip(embeddings,ori_tag_lists,spy_tag_lists):
        if not spy_tags:
            spy_vailds.append(False)
            spy_tags = [(tag,1) for tag in ori_tags]
        else:
            spy_vailds.append(True)
            
        vaild_tags = [(dv[tag[0]].tolist(),int(tag[1])) for tag in spy_tags if tag[0] in dv]
        if vaild_num:
            vaild_tags = vaild_tags[:vaild_num]
        
        vaild_tags_num = len(vaild_tags)
        if vaild_tags_num == 0:
            tag_scores.append(None)
            continue
            
        tags_embedding = np.array([x[0] for x in vaild_tags])
        tags_weights = np.array([x[1] for x in vaild_tags])
        tags_labeled_nums.append(int(np.sum(tags_weights)))
        tags_weights_ratio = tags_weights/np.sum(tags_weights)
        
        if way == 'preweight':
            tags_embedding = np.sum([x * y for x , y in zip(tags_embedding, tags_weights_ratio)], axis=0)
            scores = get_cos_similar(embedding,tags_embedding)
            tag_scores.append(scores)
        elif way == 'sum':
            scores = np.sum((get_cos_similar_multi(embedding,tags_embedding) * 2 - 1) * tags_weights)
            tag_scores.append(scores)
        else:
            raise ValueError("parm way must be premean, postmean, sum or badsum")
    assert app_num == len(tag_scores)
    return [(x,y,z) for x,y,z in zip(tag_scores,spy_vailds,tags_labeled_nums)]


def get_one_game_sentiment_score(sentiment_scores):
    mean_score = np.mean(sentiment_scores)
    if len(sentiment_scores) >= 10:
        var_score = np.var(sentiment_scores)
    else:
        var_score = None
    return mean_score,var_score


def get_one_game_feature_des(game_key_word,reviews):
    if len(game_key_word) == 0:
        return None,None,None,None,None,None,None

    game_key_titles = [x[0] for x in game_key_word]
    vaild_reviews = [x for x in reviews if len(x['key_words']) > 0]
    vaild_num = len(vaild_reviews)
    positive_vaild_num = len([x for x in vaild_reviews if int(x['voted_up']) == 1])
    mentioned_titles_dict = {}
    match_mentioned_num = 0
    match_positive_mentioned_num = 0
    match_positive_mentioned_num_xi = 0
    match_positive_mentioned_num_xi_total = 0
    for review in vaild_reviews:
        match_mentioned = False
        if_xi_positive = False
        if_xi_positive_match = False
        recommended = int(review['voted_up'])
        key_word = review['key_words']
        temp_mentioned_titles_dict = {}
        for key in key_word:
            title = key['title']
            
            if key['sentiment'] > 0.5:
                if_xi_positive = True
                if title in game_key_titles:
                    if_xi_positive_match = True
            
            if title in game_key_titles:
                match_mentioned = True
                
            if title in temp_mentioned_titles_dict.keys():
                temp_mentioned_titles_dict[title]['recommend'].append(recommended)
                temp_mentioned_titles_dict[title]['model_pred'].append(key['sentiment'])
            else:
                temp_mentioned_titles_dict[title] = {'recommend':[recommended],'model_pred':[key['sentiment']]}

        if match_mentioned:
            match_mentioned_num += 1
            if recommended == 1:
                match_positive_mentioned_num += 1
                
        if if_xi_positive:
            match_positive_mentioned_num_xi_total += 1
            if if_xi_positive_match:
                match_positive_mentioned_num_xi += 1
        
        for k,v in temp_mentioned_titles_dict.items():
            recommend_mean = np.mean(v['recommend'])
            model_pred_mean = np.mean(v['model_pred'])
            
            if k in mentioned_titles_dict.keys():
                mentioned_titles_dict[k]['recommend'].append(recommend_mean)
                mentioned_titles_dict[k]['model_pred'].append(model_pred_mean)
            else:
                mentioned_titles_dict[k] = {'recommend':[recommend_mean],'model_pred':[model_pred_mean]}
    
    vaild_match_mentioned_rate = match_mentioned_num / vaild_num if vaild_num > 0 else None
    vaild_positive_mentioned_rate = match_positive_mentioned_num / positive_vaild_num if positive_vaild_num > 0 else None
    match_positive_mentioned_rate_xi = match_positive_mentioned_num_xi / match_positive_mentioned_num_xi_total if match_positive_mentioned_num_xi_total > 0 else None
    return game_key_titles,mentioned_titles_dict,vaild_num,match_mentioned_num,vaild_match_mentioned_rate,vaild_positive_mentioned_rate,match_positive_mentioned_rate_xi


def compute_feature_scores(game_key_titles,mentioned_titles_dict):
    
    if not game_key_titles:
        return None,None,None

    game_mention_vector = np.zeros(feature_nums)
    for title in game_key_titles:
        game_mention_vector[feature_dict[title]] = 1


    user_mention_vector = np.zeros(feature_nums)
    user_mention_recomend_vector = np.zeros(feature_nums)
    user_mention_sentiment_vector = np.zeros(feature_nums)
    for k,v in mentioned_titles_dict.items():
        user_mention_vector[feature_dict[k]] = len(v['recommend'])
        user_mention_recomend_vector[feature_dict[k]] = np.mean(v['recommend'])
        user_mention_sentiment_vector[feature_dict[k]] = np.mean(v['model_pred'])

    
    game_mention_vaild_index = np.where(game_mention_vector != 0)[0]
    user_mention_vaild_index = np.where(user_mention_vector >= 5)[0]
    cross_vaild_index = np.array([x for x in game_mention_vaild_index if x in user_mention_vaild_index])
    if len(cross_vaild_index) == 0:
        sentiment_match_score = None
        sentiment_match_score_sum1 = None
        sentiment_match_score_sum2 = None
    else:
        vaild_user_mention_sentiment_vector = user_mention_sentiment_vector[cross_vaild_index]
        sentiment_match_score = np.mean(vaild_user_mention_sentiment_vector)
        sentiment_match_score_sum1 = np.sum(vaild_user_mention_sentiment_vector - 0.5)
        sentiment_match_score_sum2 = np.sum(vaild_user_mention_sentiment_vector)
    return sentiment_match_score,sentiment_match_score_sum1,sentiment_match_score_sum2


def get_timestamp(release_date):
    try:
        time_format = time.strptime(release_date, "%Y/%m/%d")
    except:
        try:
            time_format = time.strptime(release_date, "%Y/%m")
        except:
            try:
                time_format = time.strptime(release_date, "%Y")
            except:
                return -1
    return int(time.mktime(time_format))

    
def build_one_cat(cat_name,input_path,output_path,spy_tags_dict):
    print('loading apps')
    apps = []
    with open(input_path,'r') as f:
        for i in f:
            apps.append(json.loads(i))
    print('app num',len(apps))

    apps = sorted(apps,key=lambda x:int(x['game_info']['review_count']),reverse=True)
    app_time_lists = [get_timestamp(x['game_info']['release_date']) for x in apps]
    appid_lists = [x['game_info']['game_id'] for x in apps]
    sentences = [x['game_info']['app_description_short_token'] + x['game_info']['app_description_long_token'] for x in apps]
    ori_tag_lists = [x['game_info']['dv_user_tag'] for x in apps]
    spy_tag_lists = [spy_tags_dict[x] if x in spy_tags_dict.keys() else None for x in appid_lists]
    
    # model1
    tfidf_embeddings_1 = get_tfidf_embedding(sentences,model_1.wv)
    tfidf_average_socre_1,tfidf_top_score_1 = get_score(tfidf_embeddings_1,app_time_lists)
    
    tfidf_user_scores_weight_1 = get_spy_tag_score(tfidf_embeddings_1,ori_tag_lists,spy_tag_lists,model_1.dv,'preweight')
    tfidf_user_scores_sum_1 = get_spy_tag_score(tfidf_embeddings_1,ori_tag_lists,spy_tag_lists,model_1.dv,'sum')
    
                 
    #model2
    tfidf_embeddings_2 = get_tfidf_embedding(sentences,model_2.wv)

    tfidf_average_socre_2,tfidf_top_score_2 = get_score(tfidf_embeddings_2,app_time_lists)
    
    tfidf_user_scores_weight_2 = get_spy_tag_score(tfidf_embeddings_2,ori_tag_lists,spy_tag_lists,model_2.dv,'preweight')
    tfidf_user_scores_sum_2 = get_spy_tag_score(tfidf_embeddings_2,ori_tag_lists,spy_tag_lists,model_2.dv,'sum')

    
    text_compare_scores = {app['game_info']['game_id']:
                                                       {'model_1':
                                                       {'average_socre':tfidf_average_socre_1_,
                                                        'top_score':tfidf_top_score_1_,
                                                        'user_scores_weight':tfidf_user_scores_weight_1_,
                                                        'user_scores_sum':tfidf_user_scores_sum_1_,},
                                                        'model_2':
                                                       {'average_socre':tfidf_average_socre_2_,
                                                        'top_score':tfidf_top_score_2_,
                                                        'user_scores_weight':tfidf_user_scores_weight_2_,
                                                        'user_scores_sum':tfidf_user_scores_sum_2_,}}
                           
                                                           for app,
                                                           tfidf_average_socre_1_,
                                                           tfidf_top_score_1_,
                                                           tfidf_user_scores_weight_1_,
                                                           tfidf_user_scores_sum_1_,

                                                           tfidf_average_socre_2_,
                                                           tfidf_top_score_2_,
                                                           tfidf_user_scores_weight_2_,
                                                           tfidf_user_scores_sum_2_,


                                                           in zip(apps,
                                                           tfidf_average_socre_1,
                                                           tfidf_top_score_1,
                                                           tfidf_user_scores_weight_1,
                                                           tfidf_user_scores_sum_1,

                                                           tfidf_average_socre_2,
                                                           tfidf_top_score_2,
                                                           tfidf_user_scores_weight_2,
                                                           tfidf_user_scores_sum_2,)}
    
    for app in apps: 
        app_id = app['game_info']['game_id']
        review_num = len(app['game_info']['reviews'])
        if review_num < 10:
            mean_sentiment_score,var_sentiment_score = None,None
            game_key_titles,mentioned_titles_dict,vaild_num,match_mentioned_num = None,None,None,None
            vaild_match_mentioned_rate,vaild_positive_mentioned_rate,match_positive_mentioned_rate_xi = None,None,None
            rsentiment_match_score,sentiment_match_score_sum1,sentiment_match_score_sum2 = None,None,None
            
        else:
            mean_sentiment_score,var_sentiment_score = get_one_game_sentiment_score(app['game_info']['sentiments_scores'])
            game_key_titles,mentioned_titles_dict,vaild_num,match_mentioned_num,vaild_match_mentioned_rate,vaild_positive_mentioned_rate,match_positive_mentioned_rate_xi = get_one_game_feature_des(app['game_info']['game_key_word'],app['game_info']['reviews'])
            
            sentiment_match_score,sentiment_match_score_sum1,sentiment_match_score_sum2 = compute_feature_scores(game_key_titles,mentioned_titles_dict)
                
        app['variable'] = {}
        app['variable']['taget_cat_name'] = cat_name
        app['variable']['text_compare_score'] = text_compare_scores[app_id]
        app['variable']['mean_sentiment_score'] = mean_sentiment_score
        app['variable']['var_sentiment_score'] = var_sentiment_score
        
        
        app['variable']['vaild_match_mentioned_rate'] = vaild_match_mentioned_rate
        app['variable']['vaild_positive_mentioned_rate'] = vaild_positive_mentioned_rate
        app['variable']['vaild_positive_mentioned_rate_xi'] = match_positive_mentioned_rate_xi
        
        
        app['variable']['sentiment_match_score'] = sentiment_match_score
        app['variable']['sentiment_match_score_sum1'] = sentiment_match_score_sum1
        app['variable']['sentiment_match_score_sum2'] = sentiment_match_score_sum2
        
        
        app['variable']['vaild_num'] = vaild_num
        app['variable']['match_mentioned_num'] = match_mentioned_num
        app['variable']['review_num'] = review_num


        del app['game_info']['reviews']
        with open(output_path,'a+') as f:
            f.write(json.dumps(app) + '\n')
    
    print('finish')


def main(input_root,output_root):
    spy_tags_dict = {}
    with open('./spy_tags.json','r') as f:
        for i in f:
            item = json.loads(i)
            spy_tags_dict[item['game_id']] = item['dv_user_tag']
                 
    for root, dirs, files in os.walk(input_root):
        input_paths = [os.path.join(root,file) for file in files]
        output_paths = [os.path.join(output_root,file) for file in files]
        break
    
    for input_path,output_path,cat_name in tqdm(zip(input_paths,output_paths,files)):
        cat_name = cat_name.split('.')[0]
        
        if cat_name not in ['multiplayer','singleplayer','adventure']:
            print('building ',cat_name)
            build_one_cat(cat_name,input_path,output_path,spy_tags_dict)
        

if __name__ == '__main__':
    input_root = './subcat_before_embed_2018_2021'
    output_root = input_root.replace('before','after')
    if not os.path.exists(output_root):
        os.makedirs(output_root)
    main(input_root,output_root)