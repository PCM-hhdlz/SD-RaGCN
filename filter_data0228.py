# 目标：把一些时间点太小的数据集都删掉    时间点小于145的数据均不考虑
from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import numpy as np
import csv

# Input data variables
code_folder = os.getcwd()
root_folder = code_folder + '/data/'
data_folder = os.path.join(root_folder, 'ABIDE_pcp/dparsf/filt_global/')
# 如果data_folder文件夹不存在，就创建一个
if not os.path.exists(data_folder):
    os.makedirs(data_folder)
# 复制文件subject_ID.txt
shutil.copyfile(os.path.join(root_folder,'subject_ID.txt'), os.path.join(data_folder, 'subject_IDs.txt'))

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
    # argparse 用于编写用户友好的命令行接口。通过该模块，程序可以方便地从命令行读取参数
    parser = argparse.ArgumentParser(description='Download ABIDE data and compute functional connectivity matrices')
    parser.add_argument('--pipeline', default='dparsf', type=str,
                        help='Pipeline to preprocess ABIDE data. Available options are ccs, cpac, dparsf and niak.'
                             ' default: cpac.')
    parser.add_argument('--atlas', default='cc200',
                        help='Brain parcellation atlas. Options: ho, cc200 and cc400, default: cc200.')
    parser.add_argument('--download', default=True, type=str2bool,
                        help='Dowload data or just compute functional connectivity. default: True')
    args = parser.parse_args()   #parse_args()方法用于解析命令行参数。它会读取用户在命令行中输入的参数，
    print(args)

    params = dict()

    pipeline = args.pipeline
    atlas = args.atlas
    download = args.download

    # Files to fetch

    files = ['rois_' + atlas]

    filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '.1D'}


    # Download database files
    if download == True:
        # 控制下载的数据是否是fit和gobal
        abide = datasets.fetch_abide_pcp(data_dir=root_folder, pipeline=pipeline,
                                         band_pass_filtering=True, global_signal_regression=True, derivatives=files,
                                         quality_checked=False)

    subject_IDs = Reader.get_ids() #changed path to data path
    subject_IDs = subject_IDs.tolist()

    big = []  # 用于存储时间点大于等于145的样本的文件名
    small = []  # 用于存储时间点小于145的样本的文件名

    for filename in os.listdir(data_folder):
        if filename.endswith('.1D'):
            file_path = os.path.join(data_folder, filename)
            try:
                # 读取文件内容并转换为矩阵
                matrix = np.loadtxt(file_path)
                # 获取矩阵的行数
                rows = matrix.shape[0]
                if rows >= 280:
                    big.append(filename)
                else:
                    small.append(filename)
            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")

    with open('classification_result.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        # 写入表头
        writer.writerow(['类别', '文件名'])
        # 写入 big 列表的数据
        for name in big:
            writer.writerow(['X>=280', name])
        # 写入 small 列表的数据
        for name in small:
            writer.writerow(['X<280', name])

    print("分类结果已导出到 classification_result.csv 文件中。")


# 当该文件作为脚本直接运行，__name__ 自动设置为'__main__'
# 当该文件作为模块被其他文件导入时，__name__ 的值则是该模块的名称(即文件名去掉.py后缀)
if __name__ == '__main__':
    main()     # 函数调用