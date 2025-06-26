import collections

from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
import pandas as pd


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
    n_sub = 1035
    id = list(range(n_sub))

    # 读取文件
    file_path = '/root/Experiments/BrainGNN/BrainGNN_Pytorch-main/BrainGNN_Pytorch-main/Phenotypic_V1_0b_preprocessed1_1035.xlsx'
    df = pd.read_excel(file_path)

    if 'SITE_ID_NUM' in df.columns:
        # 若存在，将该列数据转换为列表
        site_id_num = df['SITE_ID_NUM'].tolist()
        print("成功获取数据SITE_ID_NUM")
    else:
        print("在 'pheno.xlsx' 文件里未找到 'SITE_ID_NUM' 列。")

    if 'DX_GROUP' in df.columns:
        # 若存在，将该列数据转换为列表
        dx_group = df['DX_GROUP'].tolist()
        print("成功获取数据DX_GROUP")
    else:
        print("在 'pheno.xlsx' 文件里未找到 'DX_GROUP' 列。")


    dic = collections.defaultdict(list)
    for i in id:
        if site_id_num[i] < 10:
            key = '0' + str(site_id_num[i]) + str(dx_group[i])
            dic[key].append(i)
        else:
            key = str(site_id_num[i]) + str(dx_group[i])
            dic[key].append(i)


    ############################################
    def split_list_randomly(lst):
        import random
        random.seed(123)
        # 随机打乱列表
        random.shuffle(lst)

        length = len(lst)
        quotient, remainder = divmod(length, 5)
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

