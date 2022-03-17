import os
from copy import deepcopy
# from tqdm import tqdm
from tqdm.auto import tqdm
import torch
import torch.nn as nn

from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs


class Trainer(object):
    """Training Helper class"""

    def __init__(self, cfg, model, data_iter, optimizer, scheduler, device):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = device[0]
        self.n_gpu = device[1]
        self.scheduler = scheduler
        self.accumulation_steps = cfg['accumulation_steps']
        
        if cfg['use_accelerator']:
            ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
            self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
            self.accelerator.wait_for_everyone()
            self.deivce = self.accelerator.device
            self.model.to(self.deivce)
            assert len(data_iter) >= 2
            if len(data_iter) == 1:
                self.model, self.optimizer, data_iter[0] = self.accelerator.prepare(
                self.model, self.optimizer, data_iter[0])
            elif len(data_iter) == 2:
                self.model, self.optimizer, data_iter[0], data_iter[1] = self.accelerator.prepare(
                self.model, self.optimizer, data_iter[0], data_iter[1])
            elif len(data_iter) == 3:
                self.model, self.optimizer, data_iter[0], data_iter[1], data_iter[2] = self.accelerator.prepare(
                self.model, self.optimizer, data_iter[0], data_iter[1], data_iter[2])
        else:
            self.model.to(self.device)
            if self.cfg['data_parallel']:  # Parallel GPU mode
                self.model = nn.DataParallel(self.model,range(self.n_gpu))
            
        # data iter
        if len(data_iter) == 1:
            self.sup_iter = data_iter[0]
        elif len(data_iter) == 2:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
        elif len(data_iter) == 3:
            self.sup_iter = self.repeat_dataloader(data_iter[0])
            self.unsup_iter = self.repeat_dataloader(data_iter[1])
            self.eval_iter = data_iter[2]

    def train(self, get_loss, get_acc):
        """ train uda"""
        self.model.train()
        
        # if not self.cfg['use_accelerator']:
        #     model = self.model.to(self.device)
        #     if self.cfg['data_parallel']:  # Parallel GPU mode
        #         model = nn.DataParallel(model,range(self.n_gpu))

        global_step = 0
        loss_sum = 0.
        max_acc = [0., 0]  # acc, step

        # Progress bar is set by unsup or sup data
        # uda_mode == True --> sup_iter is repeated
        # uda_mode == False --> sup_iter is not repeated

        iter_bar = tqdm(self.unsup_iter, total=self.cfg['total_steps']*self.accumulation_steps,
                       disable=not self.accelerator.is_local_main_process)
        
        
        des_final_loss = 0
        des_sup_loss = 0
        des_unsup_loss = 0
        des_directed_loss = 0
        des_entropy_loss = 0
        
        self.optimizer.zero_grad()
        for i, batch in enumerate(iter_bar):

            # Device assignment
            if self.cfg['use_accelerator']:
                sup_batch = next(self.sup_iter)
                unsup_batch = batch
            else:
                sup_batch = [t.to(self.device) for t in next(self.sup_iter)]
                unsup_batch = [t.to(self.device) for t in batch]                

            # update
            with self.accelerator.autocast():
                final_loss, sup_loss, unsup_loss, entropy_loss, directed_loss = get_loss(self.model, sup_batch, unsup_batch, global_step)
            
            if self.accumulation_steps > 1:
                accumulation_final_loss = final_loss / self.accumulation_steps
                
                des_final_loss += accumulation_final_loss.item()
                
                des_sup_loss += sup_loss.item()/self.accumulation_steps
                des_unsup_loss += unsup_loss.item()/self.accumulation_steps
                
                if self.cfg['un_dircted_loss']:
                    des_directed_loss += directed_loss.item()/self.accumulation_steps
                if self.cfg['entropy_min']:
                    des_entropy_loss += entropy_loss.item()/self.accumulation_steps
                    
                if self.cfg['use_accelerator']:
                    self.accelerator.backward(accumulation_final_loss)
                else:
                    accumulation_final_loss.backward()
                if (i+1) % self.accumulation_steps == 0:
                    self.optimizer.step()
                    if self.scheduler and not self.accelerator.optimizer_step_was_skipped:
                        self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1
                    iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f entropy=%5.3f dir=%5.3f' \
                                             % (des_final_loss, des_unsup_loss, des_sup_loss, des_entropy_loss, des_directed_loss))
                    des_final_loss = 0
                    des_sup_loss = 0
                    des_unsup_loss = 0
                    des_directed_loss = 0
                    des_entropy_loss = 0
                    
            else:
                if self.cfg['use_accelerator']:
                    self.accelerator.backward(final_loss)
                else:
                    final_loss.backward()
                self.optimizer.step()
                if self.scheduler:
                    self.scheduler.step()
                self.optimizer.zero_grad()

                global_step += 1
                iter_bar.set_description('final=%5.3f unsup=%5.3f sup=%5.3f dir=%5.3f' \
                                         % (final_loss.item(), unsup_loss.item(), sup_loss.item(), directed_loss.item()))
            
            loss_sum += final_loss.item()

            # logging
            # logger.add_scalars('data/scalar_group',
            #                    {'final_loss': final_loss.item(),
            #                     'sup_loss': sup_loss.item(),
            #                     'unsup_loss': unsup_loss.item(),
            #                     'lr': self.optimizer.get_lr()[0]
            #                     }, global_step)

            just_done_one_step = (i+1) % self.accumulation_steps == 0
            if global_step % self.cfg['save_steps'] == 0 and just_done_one_step:
                self.save(global_step)

            if get_acc and global_step % self.cfg['check_steps'] == 0 and global_step > self.cfg['start_checking'] and just_done_one_step:
                results = self.eval(get_acc)
                total_accuracy = torch.cat(results).mean().item()
                # logger.add_scalars('data/scalar_group', {'eval_acc': total_accuracy}, global_step)
                if max_acc[0] < total_accuracy:
                    self.save(global_step)
                    max_acc = total_accuracy, global_step
                self.accelerator.print()
                self.accelerator.print('Accuracy : %5.3f' % total_accuracy)
                self.accelerator.print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' % (
                max_acc[0], max_acc[1], global_step), end='\n\n')

            if self.cfg['total_steps'] and self.cfg['total_steps'] < global_step:
                self.accelerator.print('The total steps have been reached')
                self.accelerator.print('Average Loss %5.3f' % (loss_sum / (i + 1)))
                if get_acc:
                    results = self.eval(get_acc)
                    total_accuracy = torch.cat(results).mean().item()
                    # logger.add_scalars('data/scalar_group', {'eval_acc': total_accuracy}, global_step)
                    if max_acc[0] < total_accuracy:
                        max_acc = total_accuracy, global_step
                    self.accelerator.print()
                    self.accelerator.print('Accuracy :', total_accuracy)
                    self.accelerator.print('Max Accuracy : %5.3f Max global_steps : %d Cur global_steps : %d' % (
                    max_acc[0], max_acc[1], global_step), end='\n\n')
                self.save(global_step)
                return

        return global_step

    def eval(self, evaluate):
        """ evaluation function """
        # if model_file:
        #     self.model.eval()
        #     if not self.cfg['use_accelerator']:
        #         model = self.model.to(self.device)
        #         if self.cfg['data_parallel']:
        #             model = nn.DataParallel(model,range(self.n_gpu))
        self.model.eval()
        results = []
        iter_bar = tqdm(deepcopy(self.eval_iter),disable=not self.accelerator.is_local_main_process)
        for batch in iter_bar:
            
            if self.cfg['use_accelerator']:
                with torch.no_grad():
                    accuracy, result = evaluate(self.model, batch, self.accelerator)
            else:
                batch = [t.to(self.device) for t in batch]
                with torch.no_grad():
                    accuracy, result = evaluate(self.model, batch)
                    
            results.append(result)
            iter_bar.set_description('Eval Acc=%5.3f' % accuracy)
        self.model.train()
        return results

    def save(self, i):
        """ save model """
        if self.accelerator.is_local_main_process:
            if not os.path.isdir(os.path.join(self.cfg['results_dir'], 'save')):
                os.makedirs(os.path.join(self.cfg['results_dir'], 'save'))
            
        if self.cfg['use_accelerator']:
            self.accelerator.wait_for_everyone()
            unwrapped_model = self.accelerator.unwrap_model(self.model)
            self.accelerator.save(unwrapped_model.state_dict(), os.path.join(self.cfg['results_dir'], 'save', 'model_steps_' + str(i) + '.pth'))
            
        else:
            torch.save(self.model.state_dict(),
                       os.path.join(self.cfg['results_dir'], 'save', 'model_steps_' + str(i) + '.pt'))


    def repeat_dataloader(self, iterable):
        """ repeat dataloader """
        while True:
            for x in iterable:
                yield x
