from transformers import RobertaForMaskedLM,RobertaConfig,RobertaTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import torch
import os
from typing import Dict, List, Optional
from pyossfs.oss_bucket_manager import OSSFileManager
from torch.utils.data import Dataset

os.system(f"pip install -U git+http://gitlab.alibaba-inc.com/guohuawu.wgh/pyossfs.git")


os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,3,4,5,6,7'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print("Using {} device".format(device))

class LineByLineTextDataset(Dataset):

    def __init__(self, tokenizer, file_path, block_size):
        with OSSFileManager.open(file_path, "r") as f:
            lines = [line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())]

        batch_encoding = tokenizer(lines, add_special_tokens=True, truncation=True, max_length=block_size)
        self.examples = batch_encoding["input_ids"]
        self.examples = [{"input_ids": torch.tensor(e, dtype=torch.long)} for e in self.examples]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

model_path = "./roberta-base"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
config = RobertaConfig.from_pretrained("roberta-base")


print('build data_collator')
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
print('finish')

print('load model')
model = RobertaForMaskedLM.from_pretrained("roberta-base")
print('finish')


print('load dataset')
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path="oss://xdp-expriment/yijie.image/private/pre_train_copus_union.txt",
    #file_path="oss://xdp-expriment/yijie.image/private/test_data.txt",
    block_size=256,
)
print('finish')

training_args = TrainingArguments(
    do_train=True,
    output_dir='./',
    num_train_epochs=10,
    per_device_train_batch_size=24,
    save_strategy="no",
    logging_steps=100,
    gradient_accumulation_steps=10,
    weight_decay=0.01,
    learning_rate=2e-4,
    warmup_ratio=0.06,
    adam_epsilon=1e-6,
    adam_beta2=0.98,)

trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset,
)
print('start training')
trainer.train()
state_dict = trainer.model.state_dict()
handler = OSSFileManager.open("oss://xdp-expriment/yijie.image/private/final_model_newlr.pth", "wb")
torch.save(state_dict, handler)