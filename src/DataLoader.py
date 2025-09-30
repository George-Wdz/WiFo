# coding=utf-8
import numpy as np
import torch as th
import json
import torch
import scipy.io
import datetime
import copy
import h5py
import hdf5storage
import random
import math
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, X_train):

        self.X_train = X_train

    def __len__(self):

        return self.X_train.shape[0]

    def __getitem__(self, idx):

        return self.X_train[idx].unsqueeze(0)



def data_load_single(args, dataset): # 加载单个数据集

<<<<<<< HEAD
    folder_path_test = '../dataset/{}/X_test.mat'.format(dataset)

    X_test = hdf5storage.loadmat(folder_path_test)
    X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex)).unsqueeze(1)
=======
    folder_path_test = '../dataset/{}/X_test.mat'.format(dataset)#路径

    X_test = hdf5storage.loadmat(folder_path_test)#读.mat文件，取出x_val
    X_test_complex = torch.tensor(np.array(X_test['X_val'], dtype=complex)).unsqueeze(1)#转复数张量
>>>>>>> code-fix

    ## 实部+虚部
    X_test = torch.cat((X_test_complex.real, X_test_complex.imag),dim=1).float()

    test_data = MyDataset(X_test)


    batch_size = args.batch_size
<<<<<<< HEAD
    test_data = th.utils.data.DataLoader(test_data, num_workers=32, batch_size =  batch_size, shuffle=False, pin_memory=True, prefetch_factor=4)
=======
    test_data = th.utils.data.DataLoader(test_data, num_workers=4, batch_size =  batch_size, shuffle=False, pin_memory=True, prefetch_factor=4)
>>>>>>> code-fix

    return  test_data

def data_load(args):

    test_data_all = []

<<<<<<< HEAD
    for dataset_name in args.dataset.split('*'):
=======
    for dataset_name in args.dataset.split('*'):#支持多个数据集名字拼接，用“*”分隔
>>>>>>> code-fix
        test_data = data_load_single(args, dataset_name)
        test_data_all.append(test_data)
    
    return test_data_all


def data_load_main(args):

    test_data = data_load(args)

    return test_data

