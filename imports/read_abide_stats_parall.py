'''
Author: Xiaoxiao Li
Date: 2019/02/24
'''

import os.path as osp
from os import listdir
import os
import glob
import h5py

import torch
import numpy as np
from scipy.io import loadmat
from torch_geometric.data import Data
import networkx as nx
from networkx.convert_matrix import from_numpy_matrix
import multiprocessing
from torch_sparse import coalesce
from torch_geometric.utils import remove_self_loops
from functools import partial
import deepdish as dd
from imports.gdc import GDC
import pickle


def split(data, batch):
    node_slice = torch.cumsum(torch.from_numpy(np.bincount(batch)), 0)
    node_slice = torch.cat([torch.tensor([0]), node_slice])

    row, _ = data.edge_index_static
    edge_slice = torch.cumsum(torch.from_numpy(np.bincount(batch[row])), 0)
    edge_slice = torch.cat([torch.tensor([0]), edge_slice])

    # Edge indices should start at zero for every graph.
    data.edge_index_static -= node_slice[batch[row]].unsqueeze(0)

    # 静态和动态信息边的切分都是一致的
    slices = {'edge_index_static': edge_slice}
    slices['edge_index_dynamic'] = edge_slice

    if data.x_static is not None:
        slices['x_static'] = node_slice
        slices['x_dynamic'] = node_slice
    if data.edge_attr_static is not None:
        slices['edge_attr_static'] = edge_slice
        slices['edge_attr_dynamic'] = edge_slice
    if data.y is not None:
        if data.y.size(0) == batch.size(0):
            slices['y'] = node_slice
        else:
            slices['y'] = torch.arange(0, batch[-1] + 2, dtype=torch.long)
    if data.pos_static is not None:
        slices['pos_static'] = node_slice
        slices['pos_dynamic'] = node_slice

    return data, slices


def cat(seq):
    seq = [item for item in seq if item is not None]
    seq = [item.unsqueeze(-1) if item.dim() == 1 else item for item in seq]
    return torch.cat(seq, dim=-1).squeeze() if len(seq) > 0 else None

class NoDaemonProcess(multiprocessing.Process):
    @property
    def daemon(self):
        return False

    @daemon.setter
    def daemon(self, value):
        pass


class NoDaemonContext(type(multiprocessing.get_context())):
    Process = NoDaemonProcess


def read_data(data_dir):
    onlyfiles = [f for f in listdir(data_dir) if osp.isfile(osp.join(data_dir, f))]
    onlyfiles.sort()
    batch = []
    pseudo_static = []
    y_list = []
    edge_att_list_static, edge_index_list_static,att_list_static = [], [], []

    pseudo_dynamic = []
    edge_att_list_dynamic, edge_index_list_dynamic,att_list_dynamic = [], [], []

    # parallar computing
    cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(processes=cores)
    #pool =  MyPool(processes = cores)
    func = partial(read_sigle_data, data_dir)

    import timeit

    start = timeit.default_timer()

    res = pool.map(func, onlyfiles)

    pool.close()
    pool.join()

    # with open('res.pkl', 'wb') as file:
    #     pickle.dump(res, file)

    stop = timeit.default_timer()

    print('Time: ', stop - start)


    for j in range(len(res)):
        edge_att_list_static.append(res[j][0])         # edge_att_list 可能用于存储所有文件处理后的边属性信息
        edge_index_list_static.append(res[j][1]+j*res[j][7])  # 为了对边索引进行某种偏移处理，以区分不同文件对应的图结构
        att_list_static.append(res[j][2])                     # att_list 可能用于存储所有文件处理后的节点特征信息。
        pseudo_static.append(np.diag(np.ones(res[j][7])))     # 用于存储某种伪身份矩阵或辅助信息。对角线上元素是1，其余元素是0

        y_list.append(res[j][6])            # y_list 可能用于存储所有文件处理后的标签信息。
        batch.append([j]*res[j][7])         # batch 列表可能用于表示每个图对应的批次信息。

        edge_att_list_dynamic.append(res[j][3])
        edge_index_list_dynamic.append(res[j][4] + j * res[j][7])
        att_list_dynamic.append(res[j][5])
        temp_diag_matrix = np.diag(np.ones(res[j][7]))
        pseudo_dynamic.append(np.stack([temp_diag_matrix]*7,axis=0))

    # 将多个图的边属性、边索引、节点特征、标签等信息进行拼接和整理，然后将这些信息转换为 torch.Tensor 对象，并封装到一个 Data 对象中
    # 静态信息
    edge_att_arr_static = np.concatenate(edge_att_list_static)                  # 用于将多个数组沿指定轴拼接成一个数组，在默认轴（通常是第一个轴）上进行拼接
    edge_index_arr_static = np.concatenate(edge_index_list_static, axis=1)      # 沿第二个轴（列方向）进行拼接；edge_index_arr，用于表示所有图的边索引信息。
    att_arr_static = np.concatenate(att_list_static, axis=0)                    # 沿第一个轴（行方向）进行拼接, att_arr，用于表示所有图的节点特征信息。
    pseudo_arr_static = np.concatenate(pseudo_static, axis=0)                   # 同样指定 axis=0 进行行方向的拼接。pseudo 是一个包含多个伪身份矩阵或辅助信息矩阵的列表
    edge_att_torch_static = torch.from_numpy(edge_att_arr_static.reshape(len(edge_att_arr_static), 1)).float()      # edge_att_arr 数组重新调整形状为一个二维数组
    att_torch_static = torch.from_numpy(att_arr_static).float()                 # att_arr 数组转换为 torch.Tensor 对象，并将数据类型转换为 torch.float
    edge_index_torch_static = torch.from_numpy(edge_index_arr_static).long()    # edge_index_arr 数组转换为 torch.Tensor 对象，并将数据类型转换为 torch.long，用于表示图的边索引信息。
    pseudo_torch_static = torch.from_numpy(pseudo_arr_static).float()           # pseudo_arr 数组转换为 torch.Tensor 对象，并将数据类型转换为 torch.float

    y_arr = np.stack(y_list)                  # y_list 是一个包含多个标签数组或标签值的列表
    y_torch = torch.from_numpy(y_arr).long()  # classification         # y_arr 数组转换为 torch.Tensor 对象，并将数据类型转换为 torch.long，通常用于分类任务中的标签数据。
    batch_torch = torch.from_numpy(np.hstack(batch)).long()      # 将 batch 列表中的数组在水平方向上拼接成一个一维数组。并将数据类型转换为 torch.long

    # 动态信息
    edge_att_arr_dynamic = np.concatenate(edge_att_list_dynamic,axis=1)
    edge_index_arr_dynamic = np.concatenate(edge_index_list_dynamic,axis=2)
    att_arr_dynamic = np.concatenate(att_list_dynamic, axis=1)
    pseudo_arr_dynamic = np.concatenate(pseudo_dynamic, axis=1)
    edge_att_torch_dynamic = torch.from_numpy(edge_att_arr_dynamic.reshape(edge_att_arr_dynamic.shape[0],edge_att_arr_dynamic.shape[1], 1)).float()
    att_torch_dynamic = torch.from_numpy(att_arr_dynamic).float()
    edge_index_torch_dynamic = torch.from_numpy(edge_index_arr_dynamic).long()
    pseudo_torch_dynamic = torch.from_numpy(pseudo_arr_dynamic).float()

    # # 变换动态信息的维度
    # temp_size = att_torch_dynamic.size(1)
    # # att_torch_dynamic = att_torch_dynamic.transpose(0,1).reshape(temp_size,-1)
    # att_torch_dynamic = att_torch_dynamic.permute(1,2,0)
    #
    # temp_size1 = edge_index_torch_dynamic.size(0) * edge_index_torch_dynamic.size(1)
    # temp_size2 = edge_index_torch_dynamic.size(2)
    # # edge_index_torch_dynamic = edge_index_torch_dynamic.reshape(temp_size1,temp_size2)
    # # edge_index_torch_dynamic = edge_index_torch_dynamic.transpose(0,1)
    # edge_index_torch_dynamic = edge_index_torch_dynamic.permute(1,2,0)
    #
    # temp_size3 = edge_att_torch_dynamic.size(1)
    # # edge_att_torch_dynamic = edge_att_torch_dynamic.transpose(0,1).reshape(temp_size3,-1)
    # # edge_att_torch_dynamic = edge_att_torch_dynamic.transpose(0,1)
    # edge_att_torch_dynamic = edge_att_torch_dynamic.permute(1,2,0)
    #
    # temp_size4 = pseudo_torch_dynamic.size(1)
    # # pseudo_torch_dynamic = pseudo_torch_dynamic.transpose(0,1).reshape(temp_size4,-1)
    # # pseudo_torch_dynamic = pseudo_torch_dynamic.transpose(0,1)
    # pseudo_torch_dynamic = pseudo_torch_dynamic.permute(1,2,0)

    data = Data(x_static=att_torch_static, edge_index_static=edge_index_torch_static, y=y_torch, edge_attr_static=edge_att_torch_static, pos_static = pseudo_torch_static,
                x_dynamic=att_torch_dynamic, edge_index_dynamic=edge_index_torch_dynamic,edge_attr_dynamic=edge_att_torch_dynamic,pos_dynamic=pseudo_torch_dynamic)


    data, slices = split(data, batch_torch)

    return data, slices

# 主要作用是从指定文件中读取数据，构建图数据结构
def read_sigle_data(data_dir,filename,use_gdc =False):

    temp = dd.io.load(osp.join(data_dir, filename))

    # read edge and edge attribute
    pcorr_static = np.abs(temp['pcorr_static'][()])  # temp 数据结构中提取 'pcorr_static' 对应的数据，并计算其绝对值。
    pcorr_dynamic = np.abs(temp['pcorr_dynamic'][()])

    # 静态信息的处理
    num_nodes = pcorr_static.shape[0]    # 获取节点的数量
    G_static = from_numpy_matrix(pcorr_static)  # 函数将 pcorr 数组转换为图对象 G(它包含了图的节点和边的信息)
    A_static = nx.to_scipy_sparse_matrix(G_static)     # 图对象 G 转换为 scipy 的稀疏矩阵 A
    adj_static = A_static.tocoo()                      # 转换为 COO（Coordinate）格式的稀疏矩阵 adj
    edge_att_static = np.zeros(len(adj_static.row))
    for i in range(len(adj_static.row)):
        edge_att_static[i] = pcorr_static[adj_static.row[i], adj_static.col[i]]

    edge_index_static = np.stack([adj_static.row, adj_static.col])
    edge_index_static, edge_att_static = remove_self_loops(torch.from_numpy(edge_index_static), torch.from_numpy(edge_att_static))
    edge_index_static = edge_index_static.long()
    edge_index_static, edge_att_static = coalesce(edge_index_static, edge_att_static, num_nodes, num_nodes)

    att_static = temp['corr_static'][()]
    att_torch_static = torch.from_numpy(att_static).float()

    # 标签信息
    label = temp['label'][()]
    y_torch = torch.from_numpy(np.array(label)).long()  # classification

    # 动态信息的处理
    dynamic_frame = pcorr_dynamic.shape[0]
    edge_att_dynamic = None
    edge_index_dynamic = None
    for frame in range(dynamic_frame):
        temp_pcorr_dynamic = pcorr_dynamic[frame]
        temp_G_dynamic = from_numpy_matrix(temp_pcorr_dynamic)
        temp_A_dynamic = nx.to_scipy_sparse_matrix(temp_G_dynamic)  # 图对象 G 转换为 scipy 的稀疏矩阵 A
        temp_adj_dynamic = temp_A_dynamic.tocoo()  # 转换为 COO（Coordinate）格式的稀疏矩阵 adj

        temp_edge_att_dynamic = np.zeros(len(temp_adj_dynamic.row))

        # if edge_att_dynamic is None:
        #     edge_att_static = np.zeros(dynamic_frame,len(temp_adj_dynamic.row))
        for i in range(len(temp_adj_dynamic.row)):
            temp_edge_att_dynamic[i] = temp_pcorr_dynamic[temp_adj_dynamic.row[i], temp_adj_dynamic.col[i]]

        temp_edge_index_dynamic = np.stack([temp_adj_dynamic.row, temp_adj_dynamic.col])
        temp_edge_index_dynamic, temp_edge_att_dynamic = remove_self_loops(torch.from_numpy(temp_edge_index_dynamic),
                                                               torch.from_numpy(temp_edge_att_dynamic))
        temp_edge_index_dynamic = temp_edge_index_dynamic.long()
        temp_edge_index_dynamic, temp_edge_att_dynamic = coalesce(temp_edge_index_dynamic, temp_edge_att_dynamic, num_nodes, num_nodes)

        if edge_att_dynamic is None:
            edge_att_dynamic = torch.zeros((dynamic_frame,temp_edge_att_dynamic.shape[0]))
        if edge_index_dynamic is None:
            edge_index_dynamic = torch.zeros((dynamic_frame,temp_edge_index_dynamic.shape[0],temp_edge_index_dynamic.shape[1]))

        edge_att_dynamic[frame] = temp_edge_att_dynamic
        edge_index_dynamic[frame] = temp_edge_index_dynamic

    att_dynamic = temp['corr_dynamic'][()]
    att_torch_dynamic = torch.from_numpy(att_dynamic).float()

        #
        # edge_index_static = np.stack([adj_static.row, adj_static.col])
        # edge_index_static, edge_att_static = remove_self_loops(torch.from_numpy(edge_index_static),
        #                                                        torch.from_numpy(edge_att_static))
        # edge_index_static = edge_index_static.long()
        # edge_index_static, edge_att_static = coalesce(edge_index_static, edge_att_static, num_nodes, num_nodes)
        # att_static = temp['corr_static'][()]
        # label = temp['label'][()]
        #
        # att_torch_static = torch.from_numpy(att_static).float()
        # y_torch = torch.from_numpy(np.array(label)).long()


    data = Data(x_static=att_torch_static, edge_index_static=edge_index_static.long(), y=y_torch, edge_attr_static=edge_att_static,
                x_dynamic=att_torch_dynamic,edge_index_dynamic = edge_index_dynamic.long(),edge_attr_dynamic=edge_att_dynamic )

    if use_gdc:
        '''
        Implementation of https://papers.nips.cc/paper/2019/hash/23c894276a2c5a16470e6a31f4618d73-Abstract.html
        '''
        data.edge_attr = data.edge_attr.squeeze()
        gdc = GDC(self_loop_weight=1, normalization_in='sym',
                  normalization_out='col',
                  diffusion_kwargs=dict(method='ppr', alpha=0.2),
                  sparsification_kwargs=dict(method='topk', k=20,
                                             dim=0), exact=True)
        data = gdc(data)
        return data.edge_attr.data.numpy(),data.edge_index.data.numpy(),data.x.data.numpy(),data.y.data.item(),num_nodes

    else:
        return edge_att_static.data.numpy(),edge_index_static.data.numpy(),att_static,edge_att_dynamic.data.numpy(),edge_index_dynamic.data.numpy(),att_dynamic,label,num_nodes

if __name__ == "__main__":
    data_dir = '/home/azureuser/projects/BrainGNN/data/ABIDE_pcp/cpac/filt_noglobal/raw'
    filename = '50346.h5'
    read_sigle_data(data_dir, filename)






