import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import RobertaTokenizer,RobertaConfig

from model import Calssifier
import train
from load_data import load_data
from utils import set_seeds,get_device, _get_device, torch_device_one
from transformers import AdamW, get_linear_schedule_with_warmup

cfg = {
    "seed": 9999,
    "lr": 2e-5,
    "mode": "train_eval",
    "uda_mode": True,
    "scheduler": True,
    "num_warmup_steps": 2000,
    "total_steps": 20000,
    "max_seq_length": 512,
    "train_batch_size": 4,
    "eval_batch_size": 8,
    "data_parallel": True,
    "use_accelerator" :True,
    "accumulation_steps": 8,

    "unsup_ratio": 2,
    "uda_coeff": 1.,
    "tsa": "linear_schedule",
    "uda_softmax_temp": 1,
    "uda_confidence_thresh": -1,
    "un_dircted_loss":False,
    "dir_coeff":1.,
    "entropy_min": True,
    "entropy_coeff": 1,

    "model_path": "siebert/sentiment-roberta-large-english",

    "sup_data_dir": "data/sup_train_data.json",
    "unsup_data_dir": "data/unsup_data_list_aug_2.json",
    "eval_data_dir": "data/sup_test_data.json",

    "save_steps": 2000,
    "check_steps": 2000,
    "start_checking": 5000,
    "results_dir": "./results_2",
}

# TSA
def get_tsa_thresh(schedule, global_step, num_train_steps, start, end):
    training_progress = torch.tensor(float(global_step) / float(num_train_steps))
    if schedule == 'linear_schedule':
        threshold = training_progress
    elif schedule == 'exp_schedule':
        scale = 5
        threshold = torch.exp((training_progress - 1) * scale)
    elif schedule == 'log_schedule':
        scale = 5
        threshold = 1 - torch.exp((-training_progress) * scale)
    output = threshold * (end - start) + start
    return output


def main():
    # Load Data & Create Criterion
    print(cfg)
    model_config = RobertaConfig.from_pretrained(cfg['model_path'])
    model_config.num_labels = 5
    set_seeds(cfg['seed'])
    data = load_data(cfg)
    unsup_criterion = nn.KLDivLoss(reduction='none')
    data_iter = [data.sup_data_iter(), data.unsup_data_iter()] if cfg['mode']== 'train' \
        else [data.sup_data_iter(), data.unsup_data_iter(), data.eval_data_iter()]  # train_eval
    sup_criterion = nn.CrossEntropyLoss(reduction='none')

    # Load Model
    model = Calssifier(cfg['model_path'],model_config)

    # Create trainer
    optimizer = AdamW(model.parameters(), lr=cfg['lr'],weight_decay=0.01)
    if cfg['scheduler']:
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=cfg['num_warmup_steps'],
            num_training_steps=cfg['total_steps']
        )
    else:
        scheduler = None
    trainer = train.Trainer(cfg, model, data_iter, optimizer, scheduler, get_device())

    # Training
    def get_loss(model, sup_batch, unsup_batch, global_step):

        # logits -> prob(softmax) -> log_prob(log_softmax)
        
        # batch
        input_ids, input_mask, label_ids = sup_batch
        if unsup_batch:
            ori_input_ids, ori_input_mask, \
            aug_input_ids, aug_input_mask, un_label_ids = unsup_batch

            input_ids = torch.cat((input_ids, aug_input_ids), dim=0)
            input_mask = torch.cat((input_mask, aug_input_mask), dim=0)

        # logits
        logits = model(input_ids, input_mask, cfg['use_accelerator'])
        compute_deivce = logits.device
        # sup loss
        sup_size = label_ids.shape[0]
        sup_loss = sup_criterion(logits[:sup_size], label_ids)  # shape : train_batch_size
        if cfg['tsa']:
            tsa_thresh = get_tsa_thresh(cfg['tsa'], global_step, cfg['total_steps'], start=1. / logits.shape[-1], end=1)
            tsa_thresh = tsa_thresh.to(compute_deivce)
            larger_than_threshold = torch.exp(-sup_loss) > tsa_thresh  # prob = exp(log_prob), prob > tsa_threshold
            # larger_than_threshold = torch.sum(  F.softmax(pred[:sup_size]) * torch.eye(num_labels)[sup_label_ids]  , dim=-1) > tsa_threshold
            loss_mask = torch.ones_like(label_ids, dtype=torch.float32) * (1 - larger_than_threshold.type(torch.float32))
            sup_loss = torch.sum(sup_loss * loss_mask, dim=-1) / torch.max(torch.sum(loss_mask, dim=-1), torch_device_one(compute_deivce))
        else:
            sup_loss = torch.mean(sup_loss)

        # unsup loss
        if unsup_batch:
            # ori
            with torch.no_grad():
                ori_logits = model(ori_input_ids, ori_input_mask, cfg['use_accelerator'])
                ori_prob = F.softmax(ori_logits, dim=-1)  # KLdiv target

                # confidence-based masking
                if cfg['uda_confidence_thresh'] != -1:
                    unsup_loss_mask = torch.max(ori_prob, dim=-1)[0] > cfg['uda_confidence_thresh']
                    unsup_loss_mask = unsup_loss_mask.type(torch.float32)
                else:
                    unsup_loss_mask = torch.ones(len(logits) - sup_size, dtype=torch.float32)
                unsup_loss_mask = unsup_loss_mask.to(compute_deivce)

            # aug
            # softmax temperature controlling
            uda_softmax_temp = cfg['uda_softmax_temp'] if cfg['uda_softmax_temp'] > 0 else 1.
            
            temp_logist = logits[sup_size:] / uda_softmax_temp
            aug_log_prob = F.log_softmax(temp_logist, dim=-1)
            
            # KLdiv loss
            """
                nn.KLDivLoss (kl_div)
                input : log_prob (log_softmax)
                target : prob    (softmax)
                https://pytorch.org/docs/stable/nn.html
                unsup_loss is divied by number of unsup_loss_mask
                it is different from the google UDA official
                The official unsup_loss is divided by total
                https://github.com/google-research/uda/blob/master/text/uda.py#L175
            """
            unsup_loss = torch.sum(unsup_criterion(aug_log_prob, ori_prob), dim=-1)
            unsup_loss = torch.sum(unsup_loss * unsup_loss_mask, dim=-1) / torch.max(torch.sum(unsup_loss_mask, dim=-1),
                                                                                     torch_device_one(compute_deivce))

            
            final_loss = sup_loss + cfg['uda_coeff'] * unsup_loss
            
            # dircted loss
            if cfg['un_dircted_loss']:
                aug_logits = logits[sup_size:]
                pos_example = aug_logits[:,:2].unsqueeze(-1) - aug_logits[:,2:].unsqueeze(1)
                neg_example = aug_logits[:,3:].unsqueeze(-1) - aug_logits[:,:3].unsqueeze(1)
                pos_index = torch.where(un_label_ids == 1)[0]
                neg_index = torch.where(un_label_ids == 0)[0]
                pos_example = pos_example[pos_index]
                neg_example = neg_example[neg_index]
                all_example = torch.cat((pos_example,neg_example),dim=0)
                all_example = torch.min(all_example,dim = -1)[0]
                loss_mask = all_example > 0
                loss_mask = loss_mask.type(torch.float32).to(compute_deivce)
                vaild_expamle = len(torch.where(loss_mask==1)[0])
                if vaild_expamle == 0:
                    directed_loss = torch.tensor(0,device=compute_deivce)
                else:
                    directed_loss = all_example * loss_mask
                    directed_loss = torch.sum(directed_loss)/len(torch.where(loss_mask==1)[0])
                    final_loss += cfg['dir_coeff'] * directed_loss
                
            
            # entropy minimazation loss
            if cfg['entropy_min']:
                entropy_loss = -1.0 * F.softmax(temp_logist, dim=-1) * aug_log_prob
                entropy_loss = torch.sum(entropy_loss,dim=-1).mean()
                final_loss += cfg['entropy_coeff'] * entropy_loss
                
                
            return final_loss, sup_loss, unsup_loss, entropy_loss if cfg['entropy_min'] else None, directed_loss if cfg['un_dircted_loss'] else None

        return sup_loss, None, None, None, None

    # evaluation
    # def get_acc(model, batch, accelerator=None):
    #     input_ids, input_mask, label_id = batch
    #     logits = model(input_ids, input_mask)
    #     label_pred = logits.argmax(1)
    #     if accelerator:
    #         label_pred = accelerator.gather(label_pred)
    #         label_id = accelerator.gather(label_id)
    #     pos_label_pred = (label_pred >= 3) * 1
    #     neg_label_pred = (label_pred <= 1) * -1
    #     label_pred = pos_label_pred + neg_label_pred
    #     vaild_index = torch.where(label_pred != 0)[0]
    #     label_id = label_id[vaild_index]
    #     label_pred = label_pred[vaild_index] == 1
    #     result = (label_pred == label_id).float()
    #     accuracy = result.mean()
    #     return accuracy, result

    def get_acc(model, batch, accelerator=None):
        input_ids, input_mask, label_id = batch
        logits = model(input_ids, input_mask)
        label_pred = logits.argmax(1)
        if accelerator:
            label_pred = accelerator.gather(label_pred)
            label_id = accelerator.gather(label_id)
        result = (label_pred == label_id).float()
        accuracy = result.mean()
        return accuracy, result
    
    
    if cfg['mode'] == 'train':
        trainer.train(get_loss, None)

    if cfg['mode'] == 'train_eval':
        trainer.train(get_loss, get_acc)

if __name__ == '__main__':
    main()
