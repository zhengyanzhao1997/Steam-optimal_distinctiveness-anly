import json
from tqdm import tqdm

famous_publisher = []
with open('./famous_publisher.json','r') as f:
    for i in f:
        famous_publisher.append(i.strip('\n'))

famous_developer = []
with open('./famous_developer.json','r') as f:
    for i in f:
        famous_developer.append(i.strip('\n'))
        
        

def get_game_year(game_info,year_range):
    year_dict = {str('year_'+str(x)) : 0 for x in year_range}
    game_release_date = game_info['release_date']
    year = int(game_release_date.split('/')[0])
    year_dict['year_' + str(year)] += 1
    return year_dict


def build_price(price):
    price_dict = {'price_free':0,'price_0-10':0,'price_10-20':0,'price_20-':0}
    if price:
        price = float(price)
        if price == 0 or price == -1:
            price_dict['price_free'] = 1 
        elif 10 > price > 0:
            price_dict['price_0-10'] = 1
        elif 20 > price >= 10:
            price_dict['price_10-20'] = 1
        elif price >= 20:
            price_dict['price_20-'] = 1
    return price_dict


def build_icon(icon):
    rate_icons = {'rating_ao':0, 'rating_e':0, 'rating_e10':0, 'rating_ec':0, 'rating_m':0, 'rating_rp':0, 'rating_t':0}
    if icon:
        cat = 'rating_' + icon.split('/')[-1].split('.')[0]
        rate_icons[cat] += 1
    return rate_icons


def build_language(languages):
    language_lists = ['Arabic','Bulgarian','Czech','Danish','Dutch','English','Finnish','French','German','Greek','Hungarian',
                      'Italian','Japanese','Korean','Norwegian','Polish','Portuguese',
                      'Portuguese - Brazil','Romanian','Russian','Simplified Chinese','Slovakian','Spanish - Latin America',
                      'Spanish - Spain','Swedish','Thai','Traditional Chinese','Turkish','Ukrainian','Vietnamese']
    language_dict = {str('language_' + x):0 for x in language_lists}
    if languages:
        for lang in languages:
            if 'language_' + lang in language_dict.keys():
                language_dict['language_' + lang] = 1
    return language_dict
    
def build_feature(features):
    feature_lists = ['Remote Play on TV', 'Oculus Rift DK2', 'Steam Workshop', 'Steam Achievements', 'Includes level editor', 'Keyboard / Mouse', 'Mods (require HL2)', 'Downloadable Content', 'Stats', 'Cross-Platform Multiplayer', 'Mods', 'Single-player', 'HTC Vive', 'Standing', 'Includes Source SDK', 'Remote Play Together', 'Steam Turn Notifications', 'LAN PvP', 'SteamVR Collectibles', 'Shared/Split Screen Co-op', 'Captions available', 'Gamepad', 'MMO', 'Online PvP', 'Windows Mixed Reality', 'Partial Controller Support', 'Remote Play on Tablet', 'In-App Purchases', 'Steam Trading Cards', 'Tracked Motion Controllers', 'Room-Scale', 'Remote Play on Phone', 'Commentary available', 'Game demo', 'Oculus Rift', 'Shared/Split Screen PvP', 'Valve Anti-Cheat enabled', 'Steam Cloud', 'LAN Co-op', 'Full controller support', 'Valve Index', 'Seated', 'Steam Leaderboards', 'Online Co-op']
    feature_dict = {str('feature_' + x):0 for x in feature_lists}
    if features:
        for feature in features:
            if 'feature_' + feature in feature_dict.keys():
                feature_dict['feature_' + feature] = 1
    return feature_dict

def deal_single_game(app,year_range):
    game_info = app['game_info']
    
    muti_cat = {'if_muti_cat': 1 if len(app['subcat']) > 1 else 0}
    
    muti_developers = {'if_muti_developers': 1 if 'developer' in game_info.keys() and len(game_info['developer']) > 1 else 0}
    
    famous_dev = {'if_famous_developer' : 1 if 'developer' in game_info.keys() and len([x for x in game_info['developer'] if x in famous_developer]) > 0 else 0}
    
    muti_publishers = {'if_muti_publishers' : 1 if 'publisher' in game_info.keys() and len(game_info['publisher']) > 1 else 0}
    
    famous_pub = {'if_famous_publishers': 1 if 'publisher' in game_info.keys() and len([x for x in game_info['publisher'] if x in famous_publisher]) > 0 else 0}
    
    franch = {'if_franchiser': 1 if 'franchiser' in game_info.keys() and len(game_info['franchiser']) > 0 else 0}
    
    year = get_game_year(game_info,year_range)
    
    year = {'year':year}
    
    rating = build_icon(game_info['rating_icon'] if 'rating_icon' in game_info.keys() else None)
    
    rating = {'rating':rating}
    
    languages = build_language(game_info['language'] if 'language' in game_info.keys() else None)
    
    languages = {'languages':languages}
    
    features = build_feature(game_info['feature'] if 'feature' in game_info.keys() else None)
    
    features = {'features':features}
    
    award = {'award_bool': int(game_info['award_bool'])  if 'award_bool' in game_info.keys() else 0}
    
    hot_rate = {'hot_rate': int(game_info['recent_review_count']) / int(game_info['review_count']) if 'recent_review_count' in game_info.keys() and game_info['recent_review_count'] else 0}
    
    still_hot = {'if_still_hot': 1 if hot_rate['hot_rate'] > 1/10 else 0}
    
    muti_sys = {'if_muti_sys': 1 if "sys_card" in game_info.keys() and len(game_info["sys_card"]) > 1 else 0}
    
    DLC = {'DLC_bool': int(game_info['DLC_bool']) if 'DLC_bool' in game_info.keys() else 0}
    
    have_curator = {'curator': 1 if 'curator' in game_info.keys() and game_info["curator"] != -1 else 0}
    
    price = {'price': build_price(game_info['price'] if 'price' in game_info.keys() else None)}
    
    output_dict = {}
    output_dict.update(muti_cat)
    output_dict.update(muti_developers)
    output_dict.update(famous_dev)
    output_dict.update(muti_publishers)
    output_dict.update(famous_pub)
    output_dict.update(franch)
    output_dict.update(year)
    output_dict.update(rating)
    output_dict.update(languages)
    output_dict.update(features)
    output_dict.update(award)
    output_dict.update(hot_rate)
    output_dict.update(still_hot)
    output_dict.update(muti_sys)
    output_dict.update(DLC)
    output_dict.update(have_curator)
    output_dict.update(price)
    
    return output_dict
    
 

output_file = './after_control_220_total.json'

input_file = './final_key_words_220_total.json'


with open(output_file,'w') as f1:
    with open(input_file,'r') as f:
        for i in tqdm(f):
            app = json.loads(i)
            control = deal_single_game(app,range(2010,2023))
            app['contorl'] = control
            f1.write(json.dumps(app) + '\n')