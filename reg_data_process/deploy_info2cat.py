import json
import os
from tqdm import tqdm


# year_range = [2018,2021]
#
# dir_ = './220_subcat_before_embed_' + str(year_range[0]) + '_' + str(year_range[1]) + '/'
# if not os.path.exists(dir_):
#     os.makedirs(dir_)

total = 0
cat_dict = {}
with open('./after_control_220_total.json','r') as f:
    for i in tqdm(f):
        app = json.loads(i)
        subcat = app['subcat']
        year = app['contorl']['year']
        if year['year_2018'] > 0 or year['year_2019'] > 0 or year['year_2020'] > 0 or year['year_2021'] > 0:
            for sub in subcat:
                if sub in cat_dict.keys():
                    cat_dict[sub] += 1
                else:
                    cat_dict[sub] = 1
            total += 1
                # with open(dir_ + sub + '.json','a+') as f1:
                #     f1.write(i)

print(cat_dict)
print(total)