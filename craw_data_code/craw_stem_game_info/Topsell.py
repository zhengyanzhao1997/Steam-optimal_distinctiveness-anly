import json

all_lists = []
with open('./cat_dict.json','r') as f:
    data = json.load(f)
    for k,v in data.items():
        all_lists += [int(x) for x in v]

all_lists = set(all_lists)
print(len(all_lists))


from speed_up import main
import asyncio

new_list = [{'appid':x,'name':None} for x in all_lists]
asyncio.run(main('./cat_app_info.json',new_list))



