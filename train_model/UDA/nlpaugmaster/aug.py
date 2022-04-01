import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import torch
import os
import json
from tqdm import tqdm


#input_file = './total_detail_0_4500.json'
#output_file = './aug_0_4500_4.json'
#device_id = 0

#input_file = './total_detail_4500_9000.json'
#output_file = './aug_4500_9000_4.json'
#device_id = 1


#input_file = './total_detail_9000_13500.json'
#output_file = './aug_9000_13500_4.json'
#device_id = 2


input_file = './total_detail_13500_18000.json'
output_file = './aug_13500_18000_4.json'
device_id = 3

os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3'
device = torch.device("cuda:%s"%device_id) 
print("Using {} device".format(device))


back_translation_aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                              to_model_name='facebook/wmt19-de-en',
                                              device=device,
                                              max_length=1024)

batch_size = 16
clean_text_saved = []
voted_up_saved = []
save_count = 0
with open(input_file,'r') as f:
    for i in tqdm(f):
        app = json.loads(i)
        info = app['game_info']
        reviews = info['reviews']
        for review in reviews:
            clean_text = review['sentiment_text']
            raw_text = review['review']
            if clean_text:
                sentiment_score = review['sentiment_score']
                voted_up = review['voted_up']
                clean_text_saved.extend([clean_text]*4)
                voted_up_saved.append(voted_up)
                save_count += 1
                if save_count == batch_size:
                    augmented_text = back_translation_aug.augment(clean_text_saved)
                    assert len(augmented_text) == batch_size * 4
                    with open(output_file,'a+') as f1:
                        for index_ in range(batch_size):
                            item = {'clean_text':clean_text_saved[index_*4],
                                    'augmented_text':augmented_text[index_*4:(index_+1)*4],
                                    'voted_up':voted_up_saved[index_]}
                            f1.write(json.dumps(item) + '\n')
                    clean_text_saved = []
                    voted_up_saved = []
                    save_count = 0
