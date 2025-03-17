import os
import scipy.io as sio
import torch
from torch_geometric.data import Data
import numpy as np
from networkx.convert_matrix import from_numpy_matrix
import networkx as nx
from torch_geometric.utils import remove_self_loops
from torch_sparse import coalesce
from nilearn import connectome
from opt import *


opt = OptInit().initialize()
data_folder = opt.data_folder
id_file = "id.txt"
phenotypic_file = "phenotypic_information.csv"


def get_ids(num_subjects=None):
    """获取被试ID"""
    subject_IDs = np.genfromtxt(os.path.join(data_folder, id_file), dtype=str)
    if num_subjects is not None:
        subject_IDs = subject_IDs[:num_subjects]
    return subject_IDs


def read_sigle_data(data):
    # fc矩阵取绝对值
    pcorr = np.abs(data)
    # roi个数
    num_nodes = pcorr.shape[0]
    # 利用fc矩阵生成图
    G = from_numpy_matrix(pcorr)
    # 生成fc矩阵的 SciPy 压缩稀疏行（CSR）格式表示
    A = nx.to_scipy_sparse_array(G)
    # 将CSR稀疏表示的fc矩阵转换为coo格式
    adj = A.tocoo()
    # 获取边的属性值
    edge_att = np.zeros(len(adj.row))
    for i in range(len(adj.row)):
        edge_att[i] = pcorr[adj.row[i], adj.col[i]]
    # 获取边的索引
    edge_index = np.stack([adj.row, adj.col])
    # 去除自环边
    edge_index, edge_att = remove_self_loops(torch.from_numpy(edge_index), torch.from_numpy(edge_att))
    # 将edge_index转换成in64
    edge_index = edge_index.long()
    # 对稀疏张量进行压缩
    edge_index, edge_att = coalesce(edge_index, edge_att, num_nodes, num_nodes)
    # 处理fc矩阵,将inf值置0，再将其转为tensor.float格式
    att = data
    att[att == float('inf')] = 0
    att_torch = torch.from_numpy(att).float()
    # 将fc矩阵转换为pyg的data格式
    graph = Data(x=att_torch, edge_index=edge_index.long(), edge_attr=edge_att)
    
    return graph


def subject_connectivity(timeseries, subject, atlas_name='aal', kind='correlation', save=True, save_path=data_folder):
    """
        timeseries   : 被试的时间序列表（时间点x脑区）
        subject      : 被试ID
        atlas_name   : 用的脑区模板名称
        kind         : 要使用的连接类型, e.g. lasso, partial correlation, correlation
        save         : 是否保存
        save_path    : 保存路径

    returns:
        connectivity : 连接矩阵 (脑区 x 脑区)
    """

    print("Estimating %s matrix for subject %s" % (kind, subject))

    # 根据要计算连接的类型，计算功能矩阵
    if kind in ['tangent', 'partial correlation', 'correlation']:
        conn_measure = connectome.ConnectivityMeasure(kind=kind)
        connectivity = conn_measure.fit_transform([timeseries])[0]

    # 保存计算的功能矩阵
    if save:
        subject_file = os.path.join(save_path, subject + '_' + atlas_name + '_' + kind.replace(' ', '_') + '.mat')
        sio.savemat(subject_file, {'connectivity': connectivity})

    return connectivity


def get_timeseries(subject_list, atlas_name='aal', dataset='ABIDE'):
    """
        subject_list : 字符串格式的简短被试ID列表
        atlas_name   : 生成时间序列所基于的脑区模板名字，e.g. aal、cc200

    returns:
        time_series  : 时间序列的ndarry（样本数x时间点x脑区）
    """

    # 存储时间序列的列表，每个被试时间序列的shape为(timepoints x regions)
    timeseries = []
    # 遍历被试的列表
    for i in range(len(subject_list)):
        # 获取被试的时间序列
        if dataset == 'ABIDE':
            ro_file = [f for f in os.listdir(data_folder) if
                       f.endswith(subject_list[i] + '_rois_' + atlas_name + '.1D')]
        else:
            ro_file = [f for f in os.listdir(data_folder) if
                       f.endswith('ROISignals' + subject_list[i] + '.txt')]
        fl = os.path.join(data_folder, ro_file[0])
        print("Reading timeseries file %s" %fl)
        # 将获取的时间序列存入列表
        timeseries.append(np.loadtxt(fl, skiprows=0))

    return timeseries


def get_networks(subject_list, atlas_name="aal", kind='correlation', variable='connectivity', dataset='ABIDE'):
    graphs = []
    # 获取fc矩阵
    for subject in subject_list:
        if dataset == 'ABIDE':
            fl = os.path.join(data_folder, subject + "_" + atlas_name + "_" + kind + ".mat")
        elif dataset == 'ADNI':
            fl = os.path.join(data_folder, "ROICorrelation" + "_" + subject + ".mat")
        matrix = sio.loadmat(fl)[variable]
        with np.errstate(divide='ignore', invalid='ignore'):
            # 对fc矩阵做fisher-z标准化
            norm_matrix = np.arctanh(matrix)
        # 将fc矩阵转换为pyg的data格式
        graph = read_sigle_data(norm_matrix)               
        graphs.append(graph)

    return graphs


def get_node_feature(args):
    """获取结点特征"""
    subject_IDs = get_ids(args.num_subjects)
    raw_feature = get_networks(subject_IDs, atlas_name=args.atlas,
                               kind='correlation', variable=args.variable, dataset=args.dataset)

    return raw_feature


if __name__ == '__main__':
    subject_IDs = get_ids()

    # # 计算fc矩阵
    # time_series = get_timeseries(subject_IDs, atlas_name='aal')
    # for i in range(len(subject_IDs)):
    #     subject_connectivity(time_series[i], subject_IDs[i], atlas_name='aal')

    graphs = get_networks(subject_IDs)