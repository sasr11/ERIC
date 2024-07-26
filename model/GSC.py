from distutils.command.config import config
from platform import node
from turtle import forward
# from gpustat import print_gpustat
import time
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, GINConv, GATConv
import torch.nn.functional as F   
from torch_geometric.nn.glob import global_add_pool, global_mean_pool
from model.layers import *
from utils.gan_losses import get_negative_expectation, get_positive_expectation
from collections import OrderedDict, defaultdict
import numpy as np


class GSC_0(nn.Module):
    def __init__(self, config, n_feat):
        super(GSC, self).__init__()
        self.config                     = config
        self.n_feat                     = n_feat
        self.setup_layers()
        self.setup_score_layer()
        if config['dataset_name']== 'IMDBMulti':
            self.scale_init()

    def setup_layers(self):
        gnn_enc                         = self.config['gnn_encoder']
        self.filters                    = self.config['gnn_filters']
        self.num_filter                 = len(self.filters)  # 4
        self.use_ssl                    = self.config.get('use_ssl', False)  # True


        if self.config['fuse_type']     == 'stack':  # False
            filters                     = []
            for i in range(self.num_filter):
                filters.append(self.filters[0])
            self.filters                = filters
        self.gnn_list                   = nn.ModuleList()
        self.mlp_list_inner             = nn.ModuleList()  
        self.mlp_list_outer             = nn.ModuleList()  
        self.NTN_list                   = nn.ModuleList()

        if gnn_enc                      == 'GCN':  # append
            self.gnn_list.append(GCNConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GCNConv(self.filters[i],self.filters[i+1]))
        elif gnn_enc                    == 'GAT':
            self.gnn_list.append(GATConv(self.n_feat, self.filters[0]))
            for i in range(self.num_filter-1):   # num_filter = 3    i = 0,1   
                self.gnn_list.append(GATConv(self.filters[i],self.filters[i+1]))  
        elif gnn_enc                    == 'GIN':  # 29-64-64-32-16
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.n_feat, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ),eps=True))

            for i in range(self.num_filter-1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.filters[i],self.filters[i+1]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[i+1], self.filters[i+1]),
                torch.nn.BatchNorm1d(self.filters[i+1]),
            ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")
        # if not self.config['multi_deepsets']:
        if self.config['deepsets']:  # True
            for i in range(self.num_filter):
                if self.config['inner_mlp']:
                    if self.config.get('inner_mlp_layers', 1) == 1:  # True
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                    else:
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i], num_layers=self.config['inner_mlp_layers'], use_bn=False))
                
                if self.config.get('outer_mlp_layers', 1)     == 1:  # True
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                else:
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i], num_layers=self.config['outer_mlp_layers'], use_bn=False))
                
                self.act_inner                 = getattr(F, self.config.get('deepsets_inner_act', 'relu'))  # 设置激活函数为relu
                self.act_outer                 = getattr(F, self.config.get('deepsets_outer_act', 'relu'))
                if self.config['use_sim'] and self.config['NTN_layers'] != 1:  # False
                    self.NTN_list.append(TensorNetworkModule(self.config, self.filters[i]))
            if self.config['use_sim'] and self.config['NTN_layers'] == 1:  # True
                self.NTN                       = TensorNetworkModule(self.config, self.filters[self.num_filter-1])
            # 融合所有层的输出
            if self.config['fuse_type']        == 'cat':  # True
                self.channel_dim               = sum(self.filters)  # 所有GNN层的输出维度的总和
                self.reduction                 = self.config['reduction']  # 2
                self.conv_stack                = nn.Sequential(
                                                                nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                                                                nn.ReLU(),
                                                                nn.Dropout(p = self.config['dropout']),
                                                                nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction) ),
                                                                nn.Dropout(p = self.config['dropout']),
                                                                nn.Tanh(),
                                                            )  # 176-88-88

            elif self.config['fuse_type']      == 'stack': 
                self.conv_stack                = nn.Sequential(
                    nn.Conv1d(in_channels=3, out_channels=1, kernel_size=1),
                    nn.ReLU()
                )
                if self.config['use_sim']:
                    self.NTN                   = TensorNetworkModule(self.config, self.filters[0])
            elif self.config['fuse_type']      == 'add':
                pass
            else:
                raise RuntimeError(
                    'unsupported fuse type') 
        if self.use_ssl:
            self.GCL_model                      = GCL(self.config, sum(self.filters))
            self.gamma                          = nn.Parameter(torch.Tensor(1)) 
            
    def setup_score_layer(self):
        if self.config['deepsets']:
            # score
            if self.config['fuse_type']                  == 'cat':  # True
                self.score_layer                         = nn.Sequential(nn.Linear((self.channel_dim // self.reduction) , 16),
                                                                        nn.ReLU(),
                                                                        nn.Linear(16 , 1))
            elif self.config['fuse_type']                == 'stack': 
                self.score_layer                         = nn.Linear(self.filters[0], 1)
            # NTN
            if self.config['use_sim']:
                if self.config['NTN_layers']!=1:
                    self.score_sim_layer                 = nn.Sequential(nn.Linear(self.config['tensor_neurons'] * self.num_filter, self.config['tensor_neurons']),
                                                                        nn.ReLU(),
                                                                        nn.Linear(self.config['tensor_neurons'], 1))
                else:
                    self.score_sim_layer                 = nn.Sequential(nn.Linear(self.config['tensor_neurons'], self.config['tensor_neurons']),
                                                                        nn.ReLU(),
                                                                        nn.Linear(self.config['tensor_neurons'], 1))

        if self.config.get('output_comb', False):  # True
            self.alpha                                   = nn.Parameter(torch.Tensor(1))
            self.beta                                    = nn.Parameter(torch.Tensor(1))

    def scale_init(self):
        print("scale_init")
        nn.init.zeros_(self.gamma)
        nn.init.zeros_(self.alpha)
        nn.init.zeros_(self.beta)

    def convolutional_pass_level(self, enc, edge_index, x):
        feat                                             = enc(x, edge_index)
        feat                                             = F.relu(feat)
        feat                                             = F.dropout(feat, p = self.config['dropout'], training=self.training)
        return feat

    def deepsets_outer(self, batch, feat, filter_idx, size = None):
        size                                             = (batch[-1].item() + 1 if size is None else size)   # 一个batch中的图数
        # 按add的方式聚合每个图的节点嵌入形成图级嵌入，一共128个
        pool                                             = global_add_pool(feat, batch, size=size) if self.config['pooling']=='add' else global_mean_pool(feat, batch, size=size)
        return self.act_outer(self.mlp_list_outer[filter_idx](pool))

    def collect_embeddings(self, all_graphs):
        node_embs_dict                                   = dict()  
        for g in all_graphs:
            feat = g.x.cuda()
            edge_index = g.edge_index.cuda()
            for i, gnn in enumerate(self.gnn_list):
                if i not in node_embs_dict.keys():
                    node_embs_dict[i] = dict()
                feat                                     = gnn(feat, edge_index)  
                feat                                     = F.relu(feat)        
                node_embs_dict[i][int(g['i'])] = feat
        return node_embs_dict

    def collect_graph_embeddings(self, all_graphs):
        node_embs_dict = self.collect_embeddings(all_graphs)
        graph_embs_dicts = dict()
        for i in node_embs_dict.keys():
            if i not in graph_embs_dicts.keys():
                graph_embs_dicts[i]          = dict()  
            for k, v in node_embs_dict[i].items():   
                deepsets_inner = self.act_inner(self.mlp_list_inner[i](v))
                g_emb            = torch.sum(deepsets_inner, dim=0)
                graph_embs_dicts[i][k] = g_emb   

        return graph_embs_dicts

    def forward(self,data):
        edge_index_1            = data['g1'].edge_index.cuda()
        edge_index_2            = data['g2'].edge_index.cuda()
        features_1              = data["g1"].x.cuda()
        features_2              = data["g2"].x.cuda()
        batch_1                 = (
                                    data["g1"].batch.cuda()
                                    if hasattr(data["g1"], "batch")
                                    else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).cuda()
                                    )
        batch_2                 = (
                                    data["g2"].batch.cuda()
                                    if hasattr(data["g2"], "batch")
                                    else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).cuda()
                                    )
        
        conv_source_1           = torch.clone(features_1)
        conv_source_2           = torch.clone(features_2)
        # 训练模型or评估模式
        if not self.training:
            self.use_ssl = False
        else: self.use_ssl = True
        # 分层图卷积
        for i in range(self.num_filter):  # 分层contrast
            # print(conv_source_1.shape)
            # print(edge_index_1.shape)
            # print(batch_1.shape)
            # batc = torch.zeros(conv_source_1.size(0), dtype=torch.int64).cuda()
            # g_1 = global_add_pool(conv_source_1, batc) 
            # print(g_1)
            # print("----------------------------------------")
            conv_source_1       = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)
            conv_source_2       = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)
            if self.config['deepsets']:  # True
                # 生成图级嵌入
                if self.config.get('inner_mlp', True): # True
                    deepsets_inner_1 = self.act_inner(self.mlp_list_inner[i](conv_source_1)) # [1147, 64]
                    deepsets_inner_2 = self.act_inner(self.mlp_list_inner[i](conv_source_2))
                else:
                    deepsets_inner_1 = self.act_inner(conv_source_1)
                    deepsets_inner_2 = self.act_inner(conv_source_2)
                deepsets_outer_1     = self.deepsets_outer(batch_1, deepsets_inner_1,i)  # [128, d]  图级嵌入
                deepsets_outer_2     = self.deepsets_outer(batch_2, deepsets_inner_2,i)
                # 通过cat的方式融合每层图级嵌入间的差异
                if self.config['fuse_type']=='cat': # True
                    diff_rep         = torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat((diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2,2))), dim = 1)  
                elif self.config['fuse_type']=='stack':  # (128, 3, 1, 64)  batch_size = 128  channel  = num_filters, size= 1*64
                    diff_rep         = torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1) if i == 0 else torch.cat((diff_rep, torch.abs(deepsets_outer_1 - deepsets_outer_2).unsqueeze(1)), dim=1)   # (128,3,64)
            
                if self.config['use_sim'] and self.config['NTN_layers']!=1:
                    sim_rep          = self.NTN_list[i](deepsets_outer_1, deepsets_outer_2) if i == 0 else torch.cat((sim_rep, self.NTN_list[i](deepsets_outer_1, deepsets_outer_2)), dim = 1)  # (128, 16+16+16)
                
                if self.use_ssl:
                    cat_node_embeddings_1   = conv_source_1 if i == 0 else torch.cat((cat_node_embeddings_1, conv_source_1), dim = 1)
                    cat_node_embeddings_2   = conv_source_2 if i == 0 else torch.cat((cat_node_embeddings_2, conv_source_2), dim = 1)
                    cat_global_embedding_1  = deepsets_outer_1 if i == 0 else torch.cat((cat_global_embedding_1, deepsets_outer_1), dim = 1)
                    cat_global_embedding_2  = deepsets_outer_2 if i == 0 else torch.cat((cat_global_embedding_2, deepsets_outer_2), dim = 1)
        # AReg
        L_cl = 0
        if self.use_ssl:
            if self.config['use_deepsets']: # False
                L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2, g1 = cat_global_embedding_1, g2 = cat_global_embedding_2) * self.gamma
            else:
                L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2) * self.gamma
                print("L_cl=", L_cl.item(), "  gamma=",self.gamma.item())
            if self.config.get('cl_loss_norm', False):  # False
                if self.config.get('norm_type', 'sigmoid') == 'sigmoid':
                    L_cl = torch.sigmoid(L_cl)
                elif self.config.get('norm_type', 'sigmoid') == 'sum':
                    L_cl = torch.pow(L_cl, 2).sqrt()
                elif self.config.get('norm_type', 'sigmoid') == 'tanh':
                    L_cl = torch.tanh(L_cl)
                else:
                    raise "Norm Error"
        # 计算NTN
        if self.config['use_sim'] and self.config['NTN_layers']==1:
            sim_rep = self.NTN(deepsets_outer_1, deepsets_outer_2)
        if self.config['use_sim']:
            sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())
        # 计算score
        score_rep = self.conv_stack(diff_rep).squeeze()  # (128,64)
        score = torch.sigmoid(self.score_layer(score_rep)).view(-1)
        
        if self.config.get('use_sim', False): # True
            if self.config.get('output_comb', False): # True
                comb_score = self.alpha * score + self.beta * sim_score
                return comb_score, L_cl
            else:
                return (score + sim_score)/2 , L_cl
        else:
            return score , L_cl
        

class GSC(nn.Module):
    def __init__(self, config, n_feat):
        super(GSC, self).__init__()
        self.config                     = config
        self.n_feat                     = n_feat
        self.setup_layers()
        self.setup_score_layer()
        if config['dataset_name']== 'IMDBMulti':
            self.scale_init()

    def setup_layers(self):
        gnn_enc                         = self.config['gnn_encoder']
        self.filters                    = self.config['gnn_filters']
        self.num_filter                 = len(self.filters)  # 4
        self.use_ssl                    = self.config.get('use_ssl', False)  # True

        self.gnn_list                   = nn.ModuleList()
        self.mlp_list_inner             = nn.ModuleList()  
        self.mlp_list_outer             = nn.ModuleList()  
        self.NTN_list                   = nn.ModuleList()

        if gnn_enc                    == 'GIN':  # 29-64-64-32-16
            self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.n_feat, self.filters[0]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[0], self.filters[0]),
                torch.nn.BatchNorm1d(self.filters[0]),
            ),eps=True))

            for i in range(self.num_filter-1):
                self.gnn_list.append(GINConv(torch.nn.Sequential(
                torch.nn.Linear(self.filters[i],self.filters[i+1]),
                torch.nn.ReLU(),
                torch.nn.Linear(self.filters[i+1], self.filters[i+1]),
                torch.nn.BatchNorm1d(self.filters[i+1]),
            ), eps=True))
        else:
            raise NotImplementedError("Unknown GNN-Operator.")
        # if not self.config['multi_deepsets']:
        if self.config['deepsets']:  # True
            for i in range(self.num_filter):
                if self.config['inner_mlp']:
                    if self.config.get('inner_mlp_layers', 1) == 1:  # True
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                    else:
                        self.mlp_list_inner.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i], num_layers=self.config['inner_mlp_layers'], use_bn=False))
                
                if self.config.get('outer_mlp_layers', 1)     == 1:  # True
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i],None, num_layers=1,use_bn=False))
                else:
                    self.mlp_list_outer.append(MLPLayers(self.filters[i], self.filters[i], self.filters[i], num_layers=self.config['outer_mlp_layers'], use_bn=False))
                
                self.act_inner                 = getattr(F, self.config.get('deepsets_inner_act', 'relu'))  # 设置激活函数为relu
                self.act_outer                 = getattr(F, self.config.get('deepsets_outer_act', 'relu'))
            if self.config['use_sim'] and self.config['NTN_layers'] == 1:  # True
                self.NTN                       = TensorNetworkModule(self.config, self.filters[self.num_filter-1])
            # 融合所有层的输出
            if self.config['fuse_type']        == 'cat':  # True
                self.channel_dim               = sum(self.filters)  # 所有GNN层的输出维度的总和
                self.reduction                 = self.config['reduction']  # 2
                self.conv_stack                = nn.Sequential(
                                                                nn.Linear(self.channel_dim, self.channel_dim // self.reduction),
                                                                nn.ReLU(),
                                                                nn.Dropout(p = self.config['dropout']),
                                                                nn.Linear(self.channel_dim // self.reduction, (self.channel_dim // self.reduction) ),
                                                                nn.Dropout(p = self.config['dropout']),
                                                                nn.Tanh(),
                                                            )  # 176-88-88
            else:
                raise RuntimeError(
                    'unsupported fuse type') 
        if self.use_ssl:  # True
            self.GCL_model                      = GCL(self.config, sum(self.filters))
            self.gamma                          = nn.Parameter(torch.Tensor(1)) 
        
        # # my, cost
        self.costMatrix = GedMatrixModule(self.filters[-1], 16)  # 16-32
        # # my, LRL
        self.transform1 = torch.nn.Linear(self.filters[-1], 32)  # 16-32
        self.relu1 = torch.nn.ReLU()
        self.transform2 = torch.nn.Linear(32, 32)
        # my, 欧式距离
        # self.Eu = nn.Sequential(
        #     nn.Linear(self.filters[-1], self.filters[-1] // 2),
        #     nn.ReLU(),
        #     nn.Dropout(p = self.config['dropout']),
        #     nn.Linear(self.filters[-1] // 2, self.filters[-1] // 2),
        #     nn.Dropout(p = self.config['dropout']),
        #     nn.Tanh(),
        # )  # 16-8-8
        # self.lamda = nn.Parameter(torch.Tensor(1)) 
        
    def setup_score_layer(self):
        if self.config['deepsets']:
            # score
            if self.config['fuse_type']                  == 'cat':  # True
                self.score_layer                         = nn.Sequential(nn.Linear((self.channel_dim // self.reduction) , 16),
                                                                        nn.ReLU(),
                                                                        nn.Linear(16 , 1))
            # NTN
            if self.config['use_sim']:
                self.score_sim_layer                 = nn.Sequential(nn.Linear(self.config['tensor_neurons'], self.config['tensor_neurons']),
                                                                    nn.ReLU(),
                                                                    nn.Linear(self.config['tensor_neurons'], 1))

        if self.config.get('output_comb', False):  # True
            self.alpha                                   = nn.Parameter(torch.Tensor(1))
            self.beta                                    = nn.Parameter(torch.Tensor(1))

    def scale_init(self):
        print("scale_init")
        nn.init.zeros_(self.gamma)
        nn.init.zeros_(self.alpha)
        nn.init.zeros_(self.beta)

    def convolutional_pass_level(self, enc, edge_index, x):
        feat                                             = enc(x, edge_index)
        feat                                             = F.relu(feat)
        feat                                             = F.dropout(feat, p = self.config['dropout'], training=self.training)
        return feat

    def deepsets_outer(self, batch, feat, filter_idx, size = None):
        size                                             = (batch[-1].item() + 1 if size is None else size)   # 一个batch中的图数
        # 按add的方式聚合每个图的节点嵌入形成图级嵌入，一共128个
        pool                                             = global_add_pool(feat, batch, size=size) if self.config['pooling']=='add' else global_mean_pool(feat, batch, size=size)
        return self.act_outer(self.mlp_list_outer[filter_idx](pool))
    
    def log_sinkhorn_norm(self, log_alpha: torch.Tensor, n_iter: int =20):
        """
        行列归一化
        Args:
            log_alpha: 矩阵n*n
            n_iter: 迭代次数. Defaults to 20.
        """
        for _ in range(n_iter):
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -1, keepdim=True)
            log_alpha = log_alpha - torch.logsumexp(log_alpha, -2, keepdim=True)
        return log_alpha.exp()
    
    def gumbel_sinkhorn(self, log_alpha, tau = 1.0, n_iter = 20, noise = False, bias = None, weight = 0.5):
        """
        Args:
            log_alpha: 图对节点或边的相似度矩阵n*n
            tau: 温度系数，控制结果平滑度. Defaults to 1.0.
            n_iter: sinkhorn算法迭代次数. Defaults to 20.
            noise: 是否添加有偏置的噪声. Defaults to True.
            bias: 偏置
            weight: 偏置的权重
        Output:
            sampled_perm_mat: 采样结果
        """
        uniform_noise = torch.rand_like(log_alpha)
        gumbel_noise = -torch.log(-torch.log(uniform_noise+1e-20)+1e-20)
        if noise:
            # gumbel_noise = torch.mul(gumbel_noise, (1 + bias * weight))
            gumbel_noise = gumbel_noise + bias*0.2
        log_alpha = (log_alpha + gumbel_noise)/tau
        sampled_perm_mat = self.log_sinkhorn_norm(log_alpha, n_iter)
        return sampled_perm_mat
    
    def separate_features(self, f1, f2, batch_1, batch_2):
        """根据batch中的节点索引，分离节点向量

        Args:
            f1 (_type_): [num_nodes, d]
            f2 (_type_): [num_nodes, d]
            batch_1 (_type_): [num_nodes]  [0,0,0,1,1,2,2,....]
            batch_2 (_type_): [num_nodes]

        Returns:
            _type_: _description_
        """
        # 获取批次中图的数量
        num_graphs = batch_1.max().item() + 1
        # 创建一个列表来存储分离的特征
        separated_features_1 = [[] for _ in range(num_graphs)]
        separated_features_2 = [[] for _ in range(num_graphs)]
        # 将特征根据batch分离
        for i in range(len(batch_1)):
            graph_index = batch_1[i].item()
            separated_features_1[graph_index].append(f1[i].tolist())
        for i in range(len(batch_2)):
            graph_index = batch_2[i].item()
            separated_features_2[graph_index].append(f2[i].tolist())
        # 将列表中的特征转换为张量
        separated_features_1 = [torch.tensor(features, dtype=torch.float).cuda() for features in separated_features_1]
        separated_features_2 = [torch.tensor(features, dtype=torch.float).cuda() for features in separated_features_2]
        
        return separated_features_1, separated_features_2  # [tensor(n1,d), tensor(n2,d),...]
    
    def Cross(self, f1_list, f2_list):
        """通过嵌入计算相似度矩阵
        Args:
            Args:
            abstract_features_1 (_type_): 图1节点嵌入 [n1, 16]
            abstract_features_2 (_type_): 图2节点嵌入 [n2, 16]
        Returns:
            _type_: 堆叠的Cost矩阵(通过mask消除填充向量的影响, 同时因为之后会和置换矩阵相乘, 置换矩阵不用再mask)
            取值范围(-无穷，+无穷)
        """
        cost_matrix_list = []
        for f1, f2 in zip(f1_list, f2_list):
            n1 = f1.shape[0]
            n2 = f2.shape[0]
            max_size = 10
            cost_matrix = self.costMatrix(f1, f2)  # [n1, n2]
            # 填充成10*10的矩阵
            cost_matrix = F.pad(cost_matrix, pad=(0,max_size-n2,0,max_size-n1))  # 左右上下
            cost_matrix_list.append(cost_matrix)
            # print(n1, n2)
            # print(cost_matrix.shape)
            # print(cost_matrix)
            # print("-----------------------------------------")
            # time.sleep(2)
        return torch.stack(cost_matrix_list)  # batch*n*n
    
    def generate_CostMatrix(self, f1_list, f2_list):
        """_summary_

        Args:
            f1_list (_type_): _description_
            f2_list (_type_): _description_

        Returns:
            _type_: _description_
        """
        cost_matrix_list = []
        for f1, f2 in zip(f1_list, f2_list):
            n1 = f1.shape[0]
            n2 = f2.shape[0]
            max_size = 10
            max_n = max(n1, n2)
            f1 = F.pad(f1, pad=(0,0,0,max_n-n1))  # [max_n, 16]
            f2 = F.pad(f2, pad=(0,0,0,max_n-n2))  # [max_n, 16]
            # 计算欧式距离
            f1 = f1.unsqueeze(1)  # 扩展为 (max_n, 1, 16)
            f2 = f2.unsqueeze(0)  # 扩展为 (1, max_n, 16)
            dist = f1 - f2  # 广播后进行计算，形状为 (max_n, max_n, 16)
            dist = dist ** 2
            input = dist.view(-1, 16)  # 形状变为 (max_n*max_n, 16)
            output = self.Eu(input)  # (max_n*max_n, 8)
            output = output.view(max_n, max_n, 8)  # 将张量重塑为 (max_n, max_n, 8)
            cost_matrix = torch.sigmoid(output.sum(dim=2))  # (max_n, max_n)
            # 填充成10*10的矩阵
            cost_matrix = F.pad(cost_matrix, pad=(0,max_size-max_n,0,max_size-max_n))  # 左右上下, 
            cost_matrix_list.append(cost_matrix)
            # print(n1, n2)
            # print(cost_matrix.shape)
            # print(cost_matrix)
            # print("-----------------------------------------")
            # time.sleep(2)
        return torch.stack(cost_matrix_list)  # batch*n*n
        
    def LRL(self, f1_list, f2_list):
        """经过LRL和gumbel-sinkhorn得到置换矩阵
        Args:
            abstract_features_1 (_type_): 图1节点嵌入 [n1, 16]
            abstract_features_2 (_type_): 图2节点嵌入 [n2, 16]
        Returns:
            _type_: 堆叠的Alignment矩阵、堆叠的Cost矩阵
        """
        alignment_list = []
        for f1, f2 in zip(f1_list, f2_list):
            n1 = f1.shape[0]
            n2 = f2.shape[0]
            max_n = max(n1, n2)
            max_size = 10
            # LRL
            emb_1 = self.transform2(self.relu1(self.transform1(f1)))  # [n1, 32]
            emb_2 = self.transform2(self.relu1(self.transform1(f2)))  # [n2, 32]
            # gs
            sinkhorn_input = torch.matmul(emb_1, emb_2.permute(1,0))  # [n1, n2]
            sinkhorn_input = F.pad(sinkhorn_input, pad=(0,max_n-n2,0,max_n-n1))  # [max_n, max_n], 左右上下
            transport_plan = self.gumbel_sinkhorn(sinkhorn_input, tau=0.1)  # [max_n, max_n]
            # 填充成10*10的张量
            transport_plan = F.pad(transport_plan, pad=(0,max_size-max_n,0,max_size-max_n))  # [10, 10]
            alignment_list.append(transport_plan)
            # print(n1, n2)
            # print(sinkhorn_input.shape)
            # print(sinkhorn_input)
            # print(transport_plan.shape)
            # print(transport_plan)
            # print("-----------------------------------------")
            # time.sleep(2)
        # 虽然填充了0向量，但经过gumbel后，mask的部分仍会有值，需要cost matrix方面进行mask
        return torch.stack(alignment_list)  # batch*n*n
    
    def LRL_Cross(self, f1_list, f2_list):
        """通过节点嵌入通过LRL处理节点嵌入, 通过Cross生成Cost矩阵, 用Cost矩阵通过gs得到Alignment矩阵
        Args:
            abstract_features_1 (_type_): 图1节点嵌入 [n1, 16]
            abstract_features_2 (_type_): 图2节点嵌入 [n2, 16]
        Returns:
            _type_: 堆叠的Alignment矩阵
        """
        alignment_list = []
        cost_matrix_list = []
        for f1, f2 in zip(f1_list, f2_list):
            n1 = f1.shape[0]
            n2 = f2.shape[0]
            max_n = max(n1, n2)
            max_size = 10
            # LRL
            emb_1 = self.transform2(self.relu1(self.transform1(f1)))  # [n1, 32]
            emb_2 = self.transform2(self.relu1(self.transform1(f2)))  # [n2, 32]
            # Cross
            cost_matrix = self.costMatrix(emb_1, emb_2)  # [n1, n2]  注意costMatrix的参数
            # gs
            sinkhorn_input = F.pad(cost_matrix, pad=(0,max_n-n2,0,max_n-n1))  # [max_n, max_n]
            transport_plan = self.gumbel_sinkhorn(sinkhorn_input, tau=0.1)  # [max_n, max_n]
            # 填充
            transport_plan = F.pad(transport_plan, pad=(0,max_size-max_n,0,max_size-max_n))  # [10, 10]
            cost_matrix = F.pad(torch.exp(-torch.pow(cost_matrix, 2)), pad=(0,max_size-n2,0,max_size-n1))  # [10,10]
            alignment_list.append(transport_plan)
            cost_matrix_list.append(cost_matrix)
            # print(cost_matrix)
            # print("-------------------------------------------------------")
            # time.sleep(1)
        return torch.stack(alignment_list), torch.stack(cost_matrix_list)  # batch*n*n
    
    def forward(self,data):
        edge_index_1            = data['g1'].edge_index.cuda()
        edge_index_2            = data['g2'].edge_index.cuda()
        features_1              = data["g1"].x.cuda()
        features_2              = data["g2"].x.cuda()
        batch_1                 = (
                                    data["g1"].batch.cuda()
                                    if hasattr(data["g1"], "batch")
                                    else torch.tensor((), dtype=torch.long).new_zeros(data["g1"].num_nodes).cuda()
                                    )
        batch_2                 = (
                                    data["g2"].batch.cuda()
                                    if hasattr(data["g2"], "batch")
                                    else torch.tensor((), dtype=torch.long).new_zeros(data["g2"].num_nodes).cuda()
                                    )
        
        conv_source_1           = torch.clone(features_1)
        conv_source_2           = torch.clone(features_2)
        # 训练模型or评估模式
        if not self.training:
            self.use_ssl = False
        else: self.use_ssl = True
        self.use_ssl = False
        # 分层图卷积
        for i in range(self.num_filter):  # 分层contrast
            conv_source_1       = self.convolutional_pass_level(self.gnn_list[i], edge_index_1, conv_source_1)
            conv_source_2       = self.convolutional_pass_level(self.gnn_list[i], edge_index_2, conv_source_2)
            if self.config['deepsets']:  # True
                # 生成图级嵌入
                if self.config.get('inner_mlp', True): # True
                    deepsets_inner_1 = self.act_inner(self.mlp_list_inner[i](conv_source_1)) # [1147, 64]
                    deepsets_inner_2 = self.act_inner(self.mlp_list_inner[i](conv_source_2))
                else:
                    deepsets_inner_1 = self.act_inner(conv_source_1)
                    deepsets_inner_2 = self.act_inner(conv_source_2)
                deepsets_outer_1     = self.deepsets_outer(batch_1, deepsets_inner_1,i)  # [128, d]  图级嵌入
                deepsets_outer_2     = self.deepsets_outer(batch_2, deepsets_inner_2,i)
                # 通过cat的方式融合每层图级嵌入间的差异
                if self.config['fuse_type']=='cat': # True
                    diff_rep         = torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2, 2)) if i == 0 else torch.cat((diff_rep, torch.exp(-torch.pow(deepsets_outer_1 - deepsets_outer_2,2))), dim = 1)  
                
                
                cat_node_embeddings_1   = conv_source_1 if i == 0 else torch.cat((cat_node_embeddings_1, conv_source_1), dim = 1)
                cat_node_embeddings_2   = conv_source_2 if i == 0 else torch.cat((cat_node_embeddings_2, conv_source_2), dim = 1)
                if self.use_ssl:
                    cat_global_embedding_1  = deepsets_outer_1 if i == 0 else torch.cat((cat_global_embedding_1, deepsets_outer_1), dim = 1)
                    cat_global_embedding_2  = deepsets_outer_2 if i == 0 else torch.cat((cat_global_embedding_2, deepsets_outer_2), dim = 1)
        # AReg
        L_cl = 0
        if self.use_ssl:
            if self.config['use_deepsets']: # False
                L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2, g1 = cat_global_embedding_1, g2 = cat_global_embedding_2) * self.gamma
            else:
                L_cl = self.GCL_model(batch_1, batch_2, cat_node_embeddings_1, cat_node_embeddings_2) * self.gamma
                # print("L_cl=", L_cl.item(), "  gamma=",self.gamma.item())
        # 计算NTN
        if self.config['use_sim'] and self.config['NTN_layers']==1:
            sim_rep = self.NTN(deepsets_outer_1, deepsets_outer_2)
        if self.config['use_sim']:
            # sim_score = torch.sigmoid(self.score_sim_layer(sim_rep).squeeze())
            sim_score = self.score_sim_layer(sim_rep).squeeze()
        
        # my, 计算score
        # f1_list, f2_list = self.separate_features(cat_node_embeddings_1, cat_node_embeddings_2, batch_1, batch_2)
        # f1_list, f2_list = self.separate_features(conv_source_1, conv_source_2, batch_1, batch_2)
        # cost_matrix = self.Cross(f1_list, f2_list)  # batch*max*max
        # cost_matrix = self.generate_CostMatrix(f1_list, f2_list)  # batch*max*max
        # node_alignment = self.LRL(f1_list, f2_list)  # batch*max*max
        # node_alignment, cost_matrix = self.LRL_Cross(f1_list, f2_list)
        # soft_matrix = node_alignment * cost_matrix  # batch*max*max，逐元素相乘
        # score = torch.sigmoid(torch.sum(soft_matrix, dim=(1, 2)))  # torch.Size([batch])
        # score = torch.sum(soft_matrix, dim=(1, 2))  # torch.Size([batch])
        
        # 计算score
        score_rep = self.conv_stack(diff_rep).squeeze()  # (128,88)
        # score = torch.sigmoid(self.score_layer(score_rep)).view(-1)
        score = self.score_layer(score_rep).view(-1)
        
        if self.config.get('use_sim', False): # True
            if self.config.get('output_comb', False): # True
                comb_score = self.alpha * score + self.beta * sim_score
                # comb_score = torch.sigmoid(score + sim_score)
                print("\n",self.alpha, self.beta)
                time.sleep(0.2)
                # comb_score = self.beta * sim_score
                return comb_score, L_cl
    
class GCL(nn.Module):

    def __init__(self, config, embedding_dim):
        super(GCL, self).__init__()
        self.config        = config
        self.use_deepsets  = config['use_deepsets']
        self.use_ff        = config['use_ff']
        self.embedding_dim = embedding_dim
        self.measure       = config['measure']
        if self.use_ff:
            self.local_d   = FF(self.embedding_dim)   
            self.global_d  = FF(self.embedding_dim)
        self.init_emb()


    def init_emb(self):
        initrange = -1.5 / self.embedding_dim
        # 对神经网络中的线性层 (nn.Linear) 的权重和偏置进行初始化
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

    def forward(self, batch_1, batch_2, z1, z2, g1 = None, g2 = None):

        if not self.use_deepsets:  # True
            g1       = global_add_pool(z1, batch_1)  # num_graphs * d，batch和num_graphs等价
            g2       = global_add_pool(z2, batch_2)
        num_graphs_1 = g1.shape[0]
        num_nodes_1  = z1.shape[0]
        pos_mask_1   = torch.zeros((num_nodes_1, num_graphs_1)).cuda()
        num_graphs_2 = g2.shape[0]
        num_nodes_2  = z2.shape[0]
        pos_mask_2   = torch.zeros((num_nodes_2, num_graphs_2)).cuda()
        for node_idx, graph_idices in enumerate(zip(batch_1, batch_2)): 
            g_idx_1, g_idx_2                 = graph_idices
            pos_mask_1[node_idx][g_idx_1]    = 1.
            pos_mask_2[node_idx][g_idx_2]    = 1.
        
        if self.config.get('norm', False):
            z1 = F.normalize(z1, dim=1)
            g1 = F.normalize(g1, dim=1)
            z2 = F.normalize(z2, dim=1)
            g2 = F.normalize(g2, dim=1)
        self_sim_1   = torch.mm(z1,g1.t())   * pos_mask_1  
        self_sim_2   = torch.mm(z2,g2.t())   * pos_mask_2
        cross_sim_12 = torch.mm(z1,g2.t())   * pos_mask_1   
        cross_sim_21 = torch.mm(z2,g1.t())   * pos_mask_2
        # get_positive_expectation(self_sim_1,  self.measure, average=False)

        if self.config['sep']:  # False
            
            self_js_sim_11   = get_positive_expectation(self_sim_1,  self.measure, average=False).sum(1) 
            cross_js_sim_12  = get_positive_expectation(cross_sim_12,self.measure, average=False).sum(1)  

            self_js_sim_22   = get_positive_expectation(self_sim_2,  self.measure, average=False).sum(1)
            cross_js_sim_21  = get_positive_expectation(cross_sim_21, self.measure, average=False).sum(1)
            L_1              = (self_js_sim_11-cross_js_sim_12).pow(2).sum().sqrt()   
            L_2              = (self_js_sim_22-cross_js_sim_21).pow(2).sum().sqrt()
            return           L_1 + L_2
        else:
            L_1              = get_positive_expectation(self_sim_1,  self.measure, average=False).sum()- get_positive_expectation(cross_sim_12,self.measure, average=False).sum()
            L_2              = get_positive_expectation(self_sim_2,  self.measure, average=False).sum()- get_positive_expectation(cross_sim_21, self.measure, average=False).sum()
            # print("\nL_1=", L_1.item(), "L_2=", L_2.item())
            return           L_1 - L_2


if __name__ == "__main__":
    pass