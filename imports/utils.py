import collections
import csv
import os

import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold

import os.path as osp
from os import listdir


def train_val_test_split(kfold = 5, fold = 0):
    n_sub = 1035
    id = list(range(n_sub))


    import random
    random.seed(123)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=123,shuffle = True)
    kf2 = KFold(n_splits=kfold-1, shuffle=True, random_state = 666)


    test_index = list()
    train_index = list()
    val_index = list()

    for tr,te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id,val_id,test_id


def train_val_test_split_uniform(kfold = 10, fold = 0):
    # 指定文件夹路径
    save_folder = '/root/Experiments/SD_RaGCN/SD_RaGCN_pytorch/data/ABIDE_pcp/dparsf/filt_global/process_data_data_augmentation/raw'

    h5_files = [f for f in listdir(save_folder) if osp.isfile(osp.join(save_folder, f))]
    h5_files.sort()

    csv_dict = {}
    # 打开CSV文件
    # 读取文件
    csv_file_path = '/root/Experiments/SD_RaGCN/SD_RaGCN_pytorch/data/ABIDE_pcp/Phenotypic_V1_0b_preprocessed_site_id_num.csv'
    # Phenotypic_V1_0b_preprocessed1_site_id_num.csv
    with open(csv_file_path,'r',newline='') as csvfile:
        # 创建CSV读取器对象
        reader = csv.DictReader(csvfile)

        # 遍历CSV文件的每一行
        for row in reader:
            # 获取SUB_ID作为key
            sub_id = row['SUB_ID']
            # 获取SITE_ID_NUM和DX_GROUP作为value
            value = (row['SITE_ID_NUM'],row['DX_GROUP'])
            # 将key-value对添加到字典中
            csv_dict[sub_id] = value




    site_id_num = []
    dx_group = []

    for file in h5_files:
        file_num = file.split('.')[0]
        if len(file_num) == 6:
            file_num = file_num[:5]

        file_num_site_id_num = csv_dict[file_num][0]
        if len(file_num_site_id_num) < 2:
            file_num_site_id_num = '0' + file_num_site_id_num
        site_id_num.append(file_num_site_id_num)

        dx_group.append(csv_dict[file_num][1])


    n_sub = len(site_id_num)    # 统计受试者的数量
    id = list(range(n_sub))     # 建立一个0到n_sub范围内的列表

    dic = collections.defaultdict(list)
    for i in id:
        key = site_id_num[i] + dx_group[i]
        dic[key].append(i)


    ############################################
    def split_list_randomly(lst):
        import random
        random.seed(123)
        # 随机打乱列表
        random.shuffle(lst)

        length = len(lst)
        quotient, remainder = divmod(length, kfold)
        result = []
        start_index = 0
        for i in range(kfold):
            # 计算当前子列表的长度
            current_length = quotient + (1 if i < remainder else 0)
            # 从打乱后的列表中截取当前子列表
            sub_list = lst[start_index: start_index + current_length]
            result.append(sub_list)
            # 更新起始索引
            start_index += current_length

        # 打乱 result 中 kfold 个子列表的顺序
        random.shuffle(result)
        return result
    ############################################

    for k in dic.keys():
        dic[k] = split_list_randomly(dic[k])

    k_fold_index = [[] for _ in range(kfold)]
    for k in dic.keys():
        for j in range(kfold):
            k_fold_index[j].extend(dic[k][j])



    # test_id = k_fold_index[fold]
    # val_id = k_fold_index[kfold-1-fold]

    val_id = k_fold_index[fold]
    test_id = []

    train_id = []
    for i in range(kfold):
        # if i == fold or i == kfold-1-fold:
        #     continue
        if i == fold:
            continue
        train_id.extend(k_fold_index[i])

    test_id = np.array(test_id)
    val_id = np.array(val_id)
    train_id = np.array(train_id)

    return train_id, val_id, test_id