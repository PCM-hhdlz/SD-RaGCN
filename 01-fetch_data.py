# Copyright (c) 2019 Mwiza Kunda
# Copyright (C) 2017 Sarah Parisot <s.parisot@imperial.ac.uk>, , Sofia Ira Ktena <ira.ktena@imperial.ac.uk>
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

'''
This script mainly refers to https://github.com/kundaMwiza/fMRI-site-adaptation/blob/master/fetch_data.py
'''
import numpy as np
from nilearn import datasets
import argparse
from imports import preprocess_data as Reader
import os
import shutil
import sys

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

    params = dict()   # 创建一个空的字典

    pipeline = args.pipeline   # 访问pipeline属性
    atlas = args.atlas         # 访问atlas属性
    download = args.download   # 访问download属性

    # Files to fetch

    files = ['rois_' + atlas]

    filemapping = {'func_preproc': 'func_preproc.nii.gz',
                   files[0]: files[0] + '_new' +'.1D'}


    # Download database files
    if download == True:
        # 控制下载的数据是否是fit和gobal
        abide = datasets.fetch_abide_pcp(data_dir=root_folder, pipeline=pipeline,
                                         band_pass_filtering=True, global_signal_regression=True, derivatives=files,
                                         quality_checked=False)

    # subject_IDs = Reader.get_ids() #changed path to data path  获取了subject_IDs.txt中各个受试者的ID编号
    # subject_IDs = subject_IDs.tolist()   # numpy数组类型转换为Python列表类型

    # 数据增强
    # 定义保存文件的文件夹路径
    save_folder = '/root/Experiments/SD_RaGCN/SD_RaGCN_pytorch/data/ABIDE_pcp/dparsf/filt_global/process_data_data_augmentation/'
    # 定义存储新文件编号的列表 subject_ID_new
    subject_IDs_new = []    # 之后会把subject_ID_new存储到txt文件——subject_ID_new.txt

    # 设置取出的BOLD信号长度
    bold_length = 145

    # 遍历已经下载好.1D数据文件夹
    for filename in os.listdir(data_folder):
        if filename.endswith('.1D'):
            file_path = os.path.join(data_folder, filename)
            try:
                # 读取文件内容并转换为矩阵
                matrix = np.loadtxt(file_path)
                # 获取矩阵的行数
                rows = matrix.shape[0]
                if rows >= bold_length and rows < bold_length*2:
                    # 设置随机种子
                    np.random.seed(23)

                    # 在当前矩阵中随机取出连续的145行
                    start_index_range = rows-bold_length
                    # 随机选择一个起始行作为索引
                    start_index = np.random.randint(0, start_index_range + 1)
                    # 从原矩阵中选取从起始行开始的连续 145 行，创建新矩阵
                    matrix_new = matrix[start_index:start_index + bold_length, :]

                    # 确保保存文件夹存在，如果不存在则创建
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # 生成保存文件的文件名，这里简单以原文件名加上 _new.1D 为例
                    file_name = os.path.basename(filename).replace('.1D', '_new.1D')
                    save_path = os.path.join(save_folder, file_name)

                    # 将新矩阵保存为 .1D 文件
                    np.savetxt(save_path, matrix_new)
                    # 保存受试者编号
                    filename_part = filename.split('_')       # 将filename使用 split 方法按 _ 分割字符串
                    filename_value = filename_part[-3]        # 取出倒数第三个元素
                    filename_value = filename_value.lstrip('0')  # 去掉前面的零

                    subject_IDs_new.append(filename_value)

                elif rows >= bold_length*2:     # 做数据增强

                    # 设置随机种子
                    np.random.seed(23)

                    # 选定第一个随机矩阵的初始位置范围
                    start1_index_range = rows - bold_length*2
                    # 随机选择一个起始行作为索引
                    start1_index = np.random.randint(0, start1_index_range + 1)
                    # 从原矩阵中选取从起始行开始的连续 145 行，创建新矩阵
                    matrix_new1 = matrix[start1_index:start1_index + bold_length, :]

                    # 选定第一个随机矩阵的初始位置范围
                    start2_index_range = rows - bold_length
                    # 再随机选择一个起始行作为索引
                    start2_index = np.random.randint(start1_index+bold_length, start2_index_range + 1)
                    # 从原矩阵中选取从起始行开始的连续 145 行，创建新矩阵
                    matrix_new2 = matrix[start2_index:start2_index + bold_length, :]


                    # 确保保存文件夹存在，如果不存在则创建
                    if not os.path.exists(save_folder):
                        os.makedirs(save_folder)

                    # 保存受试者编号
                    filename_part = filename.split('_')  # 将filename使用 split 方法按 _ 分割字符串
                    filename_value = filename_part[-3]  # 取出倒数第三个元素
                    filename_value_nonzero = filename_value.lstrip('0')  # 去掉前面的零

                    # 要保存两次，因为拆分成了两个样本
                    subject_IDs_new.append(filename_value_nonzero + '1')
                    subject_IDs_new.append(filename_value_nonzero + '2')


                    # 生成保存文件的文件名，这里简单以原文件名加上 _new.1D 为例
                    filename_value1 = filename_value + '1'
                    filename_value2 = filename_value + '2'
                    file_name1 = os.path.basename(filename).replace(filename_value, filename_value1)
                    file_name2 = os.path.basename(filename).replace(filename_value, filename_value2)

                    file_name1 = os.path.basename(file_name1).replace('.1D', '_new.1D')
                    save_path1 = os.path.join(save_folder, file_name1)

                    file_name2 = os.path.basename(file_name2).replace('.1D', '_new.1D')
                    save_path2 = os.path.join(save_folder, file_name2)

                    # 将新矩阵保存为 .1D 文件
                    np.savetxt(save_path1, matrix_new1)
                    np.savetxt(save_path2, matrix_new2)



                else:   # 还有一种是受试者BOLD信号时间点小于145的，这里就不考虑了
                    continue

            except Exception as e:
                print(f"读取文件 {filename} 时出错: {e}")


    # 保存subject_IDs_new到txt文件中
    # 将每个编号转换为字符串并写入文件，后面加上换行符
    with open('/root/Experiments/SD_RaGCN/SD_RaGCN_pytorch/data/ABIDE_pcp/dparsf/filt_global/process_data_data_augmentation/subject_IDs_new.txt', 'w') as file:
        for id_num in subject_IDs_new:
            # 将每个编号转换为字符串并写入文件，后面加上换行符
            file.write(id_num + '\n')


    # Create a folder for each subject
    for s, fname in zip(subject_IDs_new, Reader.fetch_filenames(subject_IDs_new, files[0], atlas)):
        subject_folder = os.path.join(save_folder, s)
        if not os.path.exists(subject_folder):
            os.mkdir(subject_folder)

        # Get the base filename for each subject
        base = fname.split(files[0])[0]

        # Move each subject file to the subject folder
        for fl in files:
            if not os.path.exists(os.path.join(subject_folder, base + filemapping[fl])):
                shutil.move(base + filemapping[fl], subject_folder)

    time_series = Reader.get_timeseries(subject_IDs_new, atlas)

    # Compute and save connectivity matrices
    # 获取静态信息
    Reader.subject_connectivity(time_series, subject_IDs_new, atlas, 'correlation static')
    Reader.subject_connectivity(time_series, subject_IDs_new, atlas, 'partial correlation static')
    # 获取动态信息
    Reader.subject_connectivity(time_series, subject_IDs_new, atlas, 'correlation dynamic')
    Reader.subject_connectivity(time_series, subject_IDs_new, atlas, 'partial correlation dynamic')

# 当该文件作为脚本直接运行，__name__ 自动设置为'__main__'
# 当该文件作为模块被其他文件导入时，__name__ 的值则是该模块的名称(即文件名去掉.py后缀)
if __name__ == '__main__':
    main()     # 函数调用
