import json
from tqdm import tqdm

# unget_list = []
# app_file = './app_list.json'
# with open(app_file, 'r') as f:
#     appid_list = json.load(f)
#     appid_list = [x['appid'] for x in appid_list]

all_lists = []
with open('./cat_dict_new2.json','r') as f:
    data = json.load(f)
    for k,v in data.items():
        all_lists += [int(x) for x in v]

all_lists = set(all_lists)
print(len(all_lists))
# for j in tqdm(all_lists):
#     if j not in appid_list:
#         unget_list.append(j)

from speed_up import main
import asyncio

new_list = [{'appid':x,'name':None} for x in all_lists]
asyncio.run(main('./cat_app_info_new.json',new_list))

# with open('./cat_dict_set.json','w') as f:
#     for key in data.keys():
#         rank_list = data[key]
#         addr_to = list(set(rank_list))
#         addr_to.sort(key=rank_list.index)
#         new_j = {'cat_name':key,'app_lists':addr_to}
#         f.write(json.dumps(new_j)+'\n')


