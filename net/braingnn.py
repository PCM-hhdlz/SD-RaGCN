import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import TopKPooling
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
from torch_geometric.utils import (add_self_loops, sort_edge_index,
                                   remove_self_loops)
from torch_sparse import spspmm

from net.braingraphconv import MyNNConv


##########################################################################################################################
class Network(torch.nn.Module):
    def __init__(self, indim, ratio, nclass, k=8, R=200):
        '''

        :param indim: (int) node feature dimension      # 每个节点的特征维度
        :param ratio: (float) pooling ratio in (0,1)    # 池化层的池化率
        :param nclass: (int)  number of classes         # 分类的类别个数
        :param k: (int) number of communities           # 聚类数量
        :param R: (int) number of ROIs                  # 脑区数量
        '''
        super(Network, self).__init__()

        self.indim = indim
        self.dim1_s = 32
        self.dim2_s = 32
        self.dim3 = 512
        self.dim4 = 256
        self.dim5 = 8
        self.k = k
        self.R = R

        self.n1_s = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim1_s * self.indim))
        self.conv1_s = MyNNConv(self.indim, self.dim1_s, self.n1_s, normalize=False)
        self.pool1_s = TopKPooling(self.dim1_s, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)
        self.n2_s = nn.Sequential(nn.Linear(self.R, self.k, bias=False), nn.ReLU(), nn.Linear(self.k, self.dim2_s * self.dim1_s))
        self.conv2_s = MyNNConv(self.dim1_s, self.dim2_s, self.n2_s, normalize=False)
        self.pool2_s = TopKPooling(self.dim2_s, ratio=ratio, multiplier=1, nonlinearity=torch.sigmoid)

        #self.fc1 = torch.nn.Linear((self.dim2) * 2, self.dim2)
        self.fc1 = torch.nn.Linear((self.dim1_s+self.dim2_s)*2, self.dim2_s)
        self.bn1 = torch.nn.BatchNorm1d(self.dim2_s)
        self.fc2 = torch.nn.Linear(self.dim2_s, self.dim3)
        self.bn2 = torch.nn.BatchNorm1d(self.dim3)
        self.fc3 = torch.nn.Linear(self.dim3, nclass)




    def forward(self, x_s, edge_index_s, batch, edge_attr_s, pos_s):

        x_s = self.conv1_s(x_s, edge_index_s, edge_attr_s, pos_s)
        x_s, edge_index_s, edge_attr_s, batch, perm_s, score1_s = self.pool1_s(x_s, edge_index_s, edge_attr_s, batch)

        pos_s = pos_s[perm_s]
        x1_s = torch.cat([gmp(x_s, batch), gap(x_s, batch)], dim=1)

        edge_attr_s = edge_attr_s.squeeze()
        edge_index_s, edge_attr_s = self.augment_adj(edge_index_s, edge_attr_s, x_s.size(0))

        x_s = self.conv2_s(x_s, edge_index_s, edge_attr_s, pos_s)
        x_s, edge_index_s, edge_attr_s, batch, perm_s, score2_s = self.pool2_s(x_s, edge_index_s,edge_attr_s, batch)

        x2_s = torch.cat([gmp(x_s, batch), gap(x_s, batch)], dim=1)

        x_s = torch.cat([x1_s,x2_s], dim=1)
        x_s = self.bn1(F.relu(self.fc1(x_s)))
        x_s = F.dropout(x_s, p=0.5, training=self.training)
        x_s = self.bn2(F.relu(self.fc2(x_s)))
        x_s= F.dropout(x_s, p=0.5, training=self.training)
        x_s = F.log_softmax(self.fc3(x_s), dim=-1)

        return x_s,self.pool1_s.weight,self.pool2_s.weight, torch.sigmoid(score1_s).view(x_s.size(0),-1), torch.sigmoid(score2_s).view(x_s.size(0),-1)

    def augment_adj(self, edge_index, edge_weight, num_nodes):
        edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                 num_nodes=num_nodes)
        edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                  num_nodes)
        edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                         edge_weight, num_nodes, num_nodes,
                                         num_nodes)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        return edge_index, edge_weight

