import scipy.io
import torch
import torch_geometric
import numpy as np
from sklearn.model_selection import StratifiedKFold


"""
此函数是构造raw_features.pt其中data的方法 选用RV的特征 适用于WAN MAN LAN
"""
def CreateGraph_RV(matrix_path):
    # 加载.mat文件
    mat = scipy.io.loadmat(matrix_path)
    mat = mat['matrix_name']
    mat = torch.tensor(mat)
    # 将对角线上的元素置为0，只关心非对角线上的边
    mat = mat - torch.diag(torch.diag(mat))
    # 获取非零元素的坐标
    edges = torch.nonzero(mat, as_tuple=False)
    # 构造 edge_index
    edge_index = edges.t().contiguous()
    # 对 mat 取绝对值 确保生成的图中的边具有非负权重
    mat_abs = torch.abs(mat)
    # 将坐标对应的元素作为边的特征
    edge_attr = mat_abs[edge_index[0], edge_index[1]].unsqueeze(-1)
    # 将 edge_attr 转为一维
    edge_attr = edge_attr.squeeze()
    # 生成图数据
    data_RV = torch_geometric.data.Data(x=mat, edge_index=edge_index, edge_attr=edge_attr)

    return data_RV

"""
此函数是构造raw_features.pt其中data的方法 选用FC的特征 适用于功能连接矩阵
"""
def CreateGraph_FC(matrix_path):
    # 加载.mat文件
    mat = scipy.io.loadmat(matrix_path)
    mat = mat['matrix_name']
    mat = torch.tensor(mat)
    # 将对角线上的元素置为0，只关心非对角线上的边
    mat = mat - torch.diag(torch.diag(mat))
    mat = fisher_z_transform(mat)
    # 获取非零元素的坐标
    edges = torch.nonzero(mat, as_tuple=False)
    # 构造 edge_index
    edge_index = edges.t().contiguous()
    # 对 mat 取绝对值
    mat_abs = torch.abs(mat)
    # 将坐标对应的元素作为边的特征
    edge_attr = mat_abs[edge_index[0], edge_index[1]].unsqueeze(-1)
    # 将 edge_attr 转为一维
    edge_attr = edge_attr.squeeze()
    # 生成图数据
    data_FC = torch_geometric.data.Data(x=mat, edge_index=edge_index, edge_attr=edge_attr)

    return data_FC


def Create_FC_tensor(matrix_path):
    # 加载.mat文件
    mat = scipy.io.loadmat(matrix_path)
    mat = mat['matrix_name']
    FC_tensor = torch.tensor(mat)
    FC_tensor = FC_tensor - torch.diag(torch.diag(FC_tensor))
    FC_tensor_z = fisher_z_transform(FC_tensor)
    return FC_tensor_z

"""
将矩阵进行fisher_z变换
"""
def fisher_z_transform(matrix):
    with np.errstate(divide='ignore', invalid='ignore'):
        norm_matrix = np.arctanh(matrix)
    return norm_matrix


def get_node_feature(datapath):
    raw_features_fc = torch.load(datapath + '/Raw_FC_Feature.pt')
    raw_features_wan = torch.load(datapath + '/Raw_WAN_Feature.pt')
    raw_features_man = torch.load(datapath + '/Raw_MAN_Feature.pt')
    raw_features_lan = torch.load(datapath + '/Raw_LAN_Feature.pt')

    return raw_features_fc , raw_features_wan , raw_features_man , raw_features_lan

def get_node_label(datapath):
    y = torch.load(datapath + '/y_NYU.pt')

    return y

class dataloader:
    def __init__(self, args, raw_feature_fc=None, raw_feature_wan=None, raw_feature_man=None, raw_feature_lan=None, y=None, pd_dict={}):
        self.args = args
        self.seed = args.seed

        # 导入数据
        self.raw_features_fc = raw_feature_fc
        self.raw_features_wan = raw_feature_wan
        self.raw_features_man = raw_feature_man
        self.raw_features_lan = raw_feature_lan
        self.y = y
        self.pd_dict = pd_dict

    def load_data(self):
        """导入数据"""
        self.raw_features_fc = get_node_feature(self.args)
        self.raw_features_wan = get_node_feature(self.args)
        self.raw_features_man = get_node_feature(self.args)
        self.raw_features_lan = get_node_feature(self.args)
        self.y = get_node_label(self.args)

        return self.raw_features_fc , self.raw_features_wan, self.raw_features_man , self.raw_features_lan , self.y


    def data_split(self, n_folds):
        skf = StratifiedKFold(n_splits=n_folds, random_state=self.seed, shuffle=True)
        cv_splits_fc = list(skf.split(self.raw_features_fc, self.y))
        # 只要一个索引就行
        return cv_splits_fc
