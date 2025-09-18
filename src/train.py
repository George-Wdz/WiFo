# coding=utf-8
import torch
from torch.optim import AdamW, SGD, Adam
import random
import numpy as np
from sklearn.metrics import mean_squared_error,mean_absolute_error
import math
import time
import collections
import os


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
        """
        Run the model on `test_data` and compute NMSE on masked patches.

        Outputs saved to `self.args.model_path`:
        - `y_pred_{dataset}_{mask_strategy}_{mask_ratio}.npz`: flattened predicted values used to compute NMSE (masked entries only). Key: `y_pred`.
        - `y_target_{dataset}_{mask_strategy}_{mask_ratio}.npz`: flattened ground-truth values corresponding to `y_pred`. Key: `y_target`.
        - `meta_{dataset}_{mask_strategy}_{mask_ratio}.npz`: contains `patch_info` tuple needed to unpatchify.
        - `y_pred_decoded_{...}.npz`, `y_target_decoded_{...}.npz` (optional): full-dimension reconstructions obtained by `model.unpatchify` (shape: [N, T, H, W], may be complex dtype). Keys: `y_pred_decoded`, `y_target_decoded`.

        How NMSE is computed:
        For each sample in the batch we compute mean(|y_target - y_pred|^2) / mean(|y_target|^2) over the masked entries,
        then average these per-sample ratios across all samples. The code variables involved are `y_pred` and `y_target` (see below).

        How to load the saved files in Python:
        ```python
        import numpy as np
        d = np.load('y_pred_D10_temporal_0.5.npz')
        y_pred = d['y_pred']  # shape (num_samples, num_masked_points)

        d2 = np.load('y_pred_decoded_D10_temporal_0.5.npz')
        y_pred_decoded = d2['y_pred_decoded']  # shape (N, T, H, W), complex dtype possible
        ```
        """
        with torch.no_grad():
            error_nmse = 0
            num = 0

            predictions = []
            targets = []

            # lists to store reconstructed (decoded) full-dimension predictions/targets
            decoded_predictions = []
            decoded_targets = []

            patch_info = None  # 初始化 patch_info

            for _, batch in enumerate(test_data[index]):

                loss, _, pred, target, mask = self.model_forward(batch, self.model, mask_ratio, mask_strategy, seed=seed, data=dataset, mode='forward')

                if patch_info is None:
                    patch_info = self.model.patch_info  # 获取 patch_info

                dim1 = pred.shape[0]
                pred_mask = pred.squeeze(dim=2)  # [N,240,32]
                target_mask = target.squeeze(dim=2)


                y_pred = pred_mask[mask == 1].reshape(-1, 1).reshape(dim1, -1).detach().cpu().numpy()  # [Batch_size, 样本点数目]
                y_target = target_mask[mask == 1].reshape(-1, 1).reshape(dim1, -1).detach().cpu().numpy()

                predictions.append(y_pred)
                targets.append(y_target)

                # reconstruct full-dimension complex predictions/targets via model.unpatchify
                # IMPORTANT: unpatchify expects the full set of patches per sample (shape [N, L, patch_dim]).
                # Passing only the masked patches (as in `pred_mask[mask==1]`) will not reconstruct the original
                # spatial/temporal layout correctly. Use the full `pred` and `target` returned by the model.
                try:
                    # `pred` and `target` are the full patch tensors returned from model_forward
                    # shape: [N, L, patch_dim] (complex dtype)
                    pred_decoded = self.model.unpatchify(pred).detach().cpu().numpy()
                    target_decoded = self.model.unpatchify(target).detach().cpu().numpy()
                except Exception:
                    # fallback: if unpatchify fails (e.g. patch_info missing), skip decoded saving for this batch
                    pred_decoded = None
                    target_decoded = None

                if pred_decoded is not None:
                    decoded_predictions.append(pred_decoded)
                if target_decoded is not None:
                    decoded_targets.append(target_decoded)

                error_nmse += np.sum(np.mean(np.abs(y_target - y_pred) ** 2, axis=1) / np.mean(np.abs(y_target) ** 2, axis=1))
                num += y_pred.shape[0]  # 本轮mask的个数: 1000*576*0.5

            # 保存预测值和目标值
            predictions = np.concatenate(predictions, axis=0)
            targets = np.concatenate(targets, axis=0)

            output_dir = self.args.model_path
            os.makedirs(output_dir, exist_ok=True)

            pred_file = os.path.join(output_dir, f"y_pred_{dataset}_{mask_strategy}_{mask_ratio}.npz")
            target_file = os.path.join(output_dir, f"y_target_{dataset}_{mask_strategy}_{mask_ratio}.npz")
            meta_file = os.path.join(output_dir, f"meta_{dataset}_{mask_strategy}_{mask_ratio}.npz")

            np.savez(pred_file, y_pred=predictions)
            np.savez(target_file, y_target=targets)
            np.savez(meta_file, patch_info=patch_info)  # 保存 patch_info

            # 保存经过 decoder 重建回原维度的预测/目标（如果已成功重建）
            if len(decoded_predictions) > 0 and len(decoded_targets) > 0:
                decoded_predictions = np.concatenate(decoded_predictions, axis=0)
                decoded_targets = np.concatenate(decoded_targets, axis=0)

                pred_decoded_file = os.path.join(output_dir, f"y_pred_decoded_{dataset}_{mask_strategy}_{mask_ratio}.npz")
                target_decoded_file = os.path.join(output_dir, f"y_target_decoded_{dataset}_{mask_strategy}_{mask_ratio}.npz")

                # save complex arrays directly; numpy supports complex dtype in npz
                np.savez(pred_decoded_file, y_pred_decoded=decoded_predictions)
                np.savez(target_decoded_file, y_target_decoded=decoded_targets)

                print(f"Saved decoded prediction/target to: {pred_decoded_file}, {target_decoded_file}")
            else:
                print("Decoded predictions/targets were not reconstructed for any batch (unpatchify may have failed).")

           

            return error_nmse / num


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

