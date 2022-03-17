import nlpaug.augmenter.char as nac
import nlpaug.augmenter.word as naw
import nlpaug.augmenter.sentence as nas
import nlpaug.flow as nafc
from nlpaug.util import Action
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

back_translation_aug = naw.BackTranslationAug(from_model_name='facebook/wmt19-en-de',
                                              to_model_name='facebook/wmt19-de-en',
                                              device='cuda',
                                              max_length=512)

def back_translateaug(text):
    augmented_text = back_translation_aug.augment(text)
    return augmented_text