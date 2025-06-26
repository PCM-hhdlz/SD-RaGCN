import sys
import argparse
import pandas as pd
import numpy as np
from imports import preprocess_data as Reader
import deepdish as dd
import warnings
import os

# Input data variables
code_folder = os.getcwd()
root_folder = code_folder + '/data/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/dparsf/filt_global/')
# 定义保存文件的文件夹路径
save_folder = '/root/Experiments/SD_RaGCN/SD_RaGCN_pytorch/data/ABIDE_pcp/dparsf/filt_global/process_data_data_augmentation_new/'


# Process boolean command line arguments
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def main():
    # 使用 MIDA 方法来最小化 ABIDE 站点之间的分布不匹配。
    parser = argparse.ArgumentParser(description='Classification of the ABIDE dataset using a Ridge classifier. '
                                                 'MIDA is used to minimize the distribution mismatch between ABIDE sites')
    # 告知用户该参数用于网络构建的图谱选择，可用选项有 ho、cc200、cc400，默认值是 cc200。
    parser.add_argument('--atlas', default='cc200',
                        help='Atlas for network construction (node definition) options: ho, cc200, cc400, default: cc200.')
    # 该参数用于随机初始化的种子，默认值是 1234
    parser.add_argument('--seed', default=123, type=int, help='Seed for random initialisation. default: 1234.')
    # 提供帮助信息，说明该参数表示分类的类别数量，默认值是 2。
    parser.add_argument('--nclass', default=2, type=int, help='Number of classes. default:2')

    args = parser.parse_args()      # 解析命令行中用户输入的参数
    print('Arguments: \n', args)    # 将解析后的命令行参数信息打印输出，方便用户查看脚本接收到的参数情况。

    params = dict()
    params['seed'] = args.seed  # seed for random initialisation
    # Algorithm choice
    params['atlas'] = args.atlas  # Atlas for network construction
    atlas = args.atlas  # Atlas for network construction (node definition)

    # Get subject IDs and class labels
    subject_IDs = Reader.get_ids()
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')

    # Number of subjects and classes for binary classification
    num_classes = args.nclass
    num_subjects = len(subject_IDs)
    params['n_subjects'] = num_subjects

    # Initialise variables for class labels and acquisition sites
    # 1 is autism, 2 is control
    y_data = np.zeros([num_subjects, num_classes])  # n x 2
    y = np.zeros([num_subjects, 1])  # n x 1

    # Get class labels for all subjects
    for i in range(num_subjects):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1
        y[i] = int(labels[subject_IDs[i]])

    # Compute feature vectors (vectorised connectivity networks)
    fea_corr_static = Reader.get_networks(subject_IDs, iter_no='', kind='correlation static', atlas_name=atlas)  # (1035, 200, 200)
    fea_pcorr_static = Reader.get_networks(subject_IDs, iter_no='', kind='partial correlation static', atlas_name=atlas)  # (1035, 200, 200)

    fea_corr_dynamic = Reader.get_networks(subject_IDs, iter_no='', kind='correlation dynamic',atlas_name=atlas)  # (1035, 200, 200)
    fea_pcorr_dynamic = Reader.get_networks(subject_IDs, iter_no='', kind='partial correlation dynamic',atlas_name=atlas)  # (1035, 200, 200)

    if not os.path.exists(os.path.join(save_folder,'raw')):
        os.makedirs(os.path.join(save_folder,'raw'))
    for i, subject in enumerate(subject_IDs):
        dd.io.save(os.path.join(save_folder,'raw',subject+'.h5'),{'corr_static':fea_corr_static[i],'pcorr_static':fea_pcorr_static[i],'corr_dynamic':fea_corr_dynamic[i],'pcorr_dynamic':fea_pcorr_dynamic[i],'label':y[i]%2})

if __name__ == '__main__':
    main()