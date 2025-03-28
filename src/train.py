# coding=utf-8
import torch
from torch.optim import AdamW, SGD, Adam
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
import collections


class TrainLoop:
    def __init__(self, args, writer, model, test_data, device, early_stop = 5):
        self.args = args
        self.writer = writer
        self.model = model
        self.test_data = test_data
        self.device = device
        self.lr_anneal_steps = args.lr_anneal_steps
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.opt = AdamW([p for p in self.model.parameters() if p.requires_grad==True], lr=args.lr, weight_decay=args.weight_decay)
        self.log_interval = args.log_interval
        self.best_nmse_random = 1e9
        self.warmup_steps=5
        self.min_lr = args.min_lr
        self.best_nmse = 1e9
        self.early_stop = early_stop
        
        self.mask_list = {'random':[0.85],'temporal':[0.5], 'fre':[0.5]}


    def Sample(self, test_data, step, mask_ratio, mask_strategy, seed=None, dataset='', index=0):
        with torch.no_grad():
            error_nmse = 0
            num=0
            # start time
            for _, batch in enumerate(test_data[index]):

                loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data = dataset, mode='forward')


                dim1 = pred.shape[0]
                pred_mask = pred.squeeze(dim=2)  # [N,240,32]
                target_mask = target.squeeze(dim=2)


                y_pred = pred_mask[mask==1].reshape(-1,1).reshape(dim1,-1).detach().cpu().numpy()  # [Batch_size, 样本点数目]
                y_target = target_mask[mask==1].reshape(-1,1).reshape(dim1,-1).detach().cpu().numpy()

                error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                num += y_pred.shape[0]  # 本轮mask的个数: 1000*576*0.5

        nmse = error_nmse / num

        return nmse


    def Evaluation(self, test_data, epoch, seed=None):


        nmse_list = []
        nmse_key_result = {}

        for index, dataset_name in enumerate(self.args.dataset.split('*')):

            nmse_key_result[dataset_name] = {}

            if self.args.mask_strategy_random != 'none':
                mask_list = self.mask_list_chosen(dataset_name)  # 自定义mask_list
                for s in mask_list:
                    for m in self.mask_list[s]:
                        nmse = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index)
                        nmse_list.append(nmse)
                        if s not in nmse_key_result[dataset_name]:
                            nmse_key_result[dataset_name][s] = {}
                        nmse_key_result[dataset_name][s][m] = nmse
                        

                        self.writer.add_scalar('Test_NMSE/{}-{}-{}'.format(dataset_name.split('_C')[0], s, m), nmse, epoch)

            else:
                s = self.args.mask_strategy
                m = self.args.mask_ratio
                nmse = self.Sample(test_data, epoch, mask_ratio=m, mask_strategy = s, seed=seed, dataset = dataset_name, index=index)
                nmse_list.append(nmse)
                if s not in nmse_key_result[dataset_name]:
                    nmse_key_result[dataset_name][s] = {}
                nmse_key_result[dataset_name][s][m] = {'nmse':nmse}


                self.writer.add_scalar('Test_NMSE/Stage-{}-{}-{}-{}'.format(self.args.stage, dataset_name.split('_C')[0], s, m), nmse, epoch)

        
        loss_test = np.mean(nmse_list)

        is_break = self.best_model_save(epoch, loss_test, nmse_key_result)
        return is_break  # 输出的是“save”

    def best_model_save(self, step, nmse, nmse_key_result):

        self.early_stop = 0
        torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best_stage_{}.pkl'.format(self.args.stage))
        torch.save(self.model.state_dict(), self.args.model_path+'model_save/model_best.pkl')
        self.best_nmse = nmse
        self.writer.add_scalar('Evaluation/NMSE_best', self.best_nmse, step)
        print('\nNMSE_best:{}\n'.format(self.best_nmse))
        print(str(nmse_key_result) + '\n')
        with open(self.args.model_path+'result.txt', 'w') as f:
            f.write('stage:{}, epoch:{}, best nmse: {}\n'.format(self.args.stage, step, self.best_nmse))
            f.write(str(nmse_key_result) + '\n')
        with open(self.args.model_path+'result_all.txt', 'a') as f:
            f.write('stage:{}, epoch:{}, best nmse: {}\n'.format(self.args.stage, step, self.best_nmse))
            f.write(str(nmse_key_result) + '\n')
        return 'save'

    def mask_select(self,name):
        if self.args.mask_strategy_random == 'none': #'none' or 'batch'
            mask_strategy = self.args.mask_strategy
            mask_ratio = self.args.mask_ratio
        else:
            mask_strategy = random.choice(['random','temporal','fre'])
            mask_ratio = random.choice(self.mask_list[mask_strategy])

        return mask_strategy, mask_ratio


    def mask_list_chosen(self,name):
        if self.args.mask_strategy_random == 'none': #'none' or 'batch'
            mask_list = self.mask_list
        else:
            mask_list = {key: self.mask_list[key] for key in ['random','temporal','fre']}
        return mask_list

    def run_loop(self):

        self.Evaluation(self.test_data, 0)

    def model_forward(self, batch, model, mask_ratio, mask_strategy, seed=None, data=None, mode='backward'):

        batch = [i.to(self.device) for i in batch]

        loss, loss2, pred, target, mask = self.model(
                batch,
                mask_ratio=mask_ratio,
                mask_strategy = mask_strategy, 
                seed = seed, 
                data = data,
            )
        return loss, loss2, pred, target, mask

