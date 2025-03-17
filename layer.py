"""
Author:YueYang Li
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import ChebConv
import warnings
warnings.filterwarnings("ignore")

from opt import *
from utils.utils import dataloader
opt = OptInit().initialize()

def add_dimensions(tensor):
    # 使用unsqueeze函数增加新的维度
    tensor = tensor.unsqueeze(1)  # 在第1维增加一个新的维度
    return tensor


def upper_triangle_concat(matrix):
    matrix = [mat.numpy() for mat in matrix]

    # 获取上三角元素
    idx = np.triu_indices_from(matrix[0], 1)
    upper_triangle = [mat[idx] for mat in matrix]
    upper_triangle = torch.tensor(np.vstack(upper_triangle)).float()

    return upper_triangle


"C-Stream"
class MatrixToFeature(nn.Module):
    def __init__(self):
        super(MatrixToFeature, self).__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(16, 1, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(30135, 128)

    def forward(self, x):
        # # 添加batch维度并调整为(1, L)的形状
        x = torch.tensor(x, dtype=torch.float32).view(-1, 1, 30135).to(opt.device)
        # 一维卷积层
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        # 展平
        x = x.view(x.size(0), -1)
        # MLP
        x = F.relu(self.fc1(x))
        return x


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.conv = nn.Sequential(
            # 1
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 2
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            # 3
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 4
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 5
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            # 6
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 7
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 8
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            # 9
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 10
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 11
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 12
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # 13
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 14
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            # 15
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.AvgPool2d(kernel_size=1, stride=1),
        )
        self.fc = nn.Sequential(
            # 16
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 17
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            # 18
            nn.Linear(4096, 1024),
            nn.ReLU(True),
            nn.Dropout(),
            # 19
            nn.Linear(1024, 512),
        )


    def forward(self, x):
        x = add_dimensions(x).to(opt.device)
        x = x.float()
        feature = self.conv(x)
        feature = feature.view(feature.size(0), -1)

        VGG_embedding = self.fc(feature)

        return VGG_embedding


"G-Stream"
"ChebNet_WAN"
class ChebNet_WAN(nn.Module):
    def __init__(self):
        super(ChebNet_WAN, self).__init__()
        self.conv1 = ChebConv(in_channels=7, out_channels=6, K=2)
        self.conv2 = ChebConv(in_channels=6, out_channels=6, K=2)
        self.conv3 = ChebConv(in_channels=6, out_channels=6, K=2)
        self.bn1 = nn.BatchNorm1d(6)

        self.fc1 = nn.Linear(42, 16)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.float()
        edge_attr = edge_attr.float()
        layer_out = []
        x = F.relu(self.conv1(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(batch[-1] + 1, -1)

        graph_embedding = self.fc1(x)

        return graph_embedding

"ChebNet_MAN"
class ChebNet_MAN(nn.Module):
    def __init__(self):
        super(ChebNet_MAN, self).__init__()
        self.conv1 = ChebConv(in_channels=26, out_channels=20, K=2)
        self.conv2_0 = ChebConv(in_channels=20, out_channels=20, K=2)
        self.conv2_1 = ChebConv(in_channels=20, out_channels=20, K=2)
        self.conv3 = ChebConv(in_channels=20, out_channels=20, K=2)
        self.bn1 = nn.BatchNorm1d(20)
        self.fc1 = nn.Linear(20*26, 128)

    def forward(self, x, edge_index, edge_attr, batch):
        # 处理数据
        x = x.float()
        edge_index = edge_index.view(2, -1)
        edge_attr = edge_attr.float()
        layer_out = []
        x = F.relu(self.conv1(x, edge_index, edge_attr, batch=batch)).to(opt.device)
        x = self.bn1(x)
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2_0(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2_1(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[2]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(batch[-1] + 1, -1)

        graph_embedding = self.fc1(x)

        return graph_embedding

class ChebNet(nn.Module):
    def __init__(self):
        super(ChebNet, self).__init__()
        self.conv1 = ChebConv(in_channels=207, out_channels=128, K=2)
        self.conv2_0 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.conv2_1 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.conv2_2 = ChebConv(in_channels=128, out_channels=128, K=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv3 = ChebConv(in_channels=128, out_channels=64, K=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv4 = ChebConv(in_channels=64, out_channels=8, K=2)
        self.bn3 = nn.BatchNorm1d(8)
        self.fc1 = nn.Linear(1656, 128)

    def forward(self, x, edge_index, edge_attr, batch):
        # 处理数据
        x = x.float()
        edge_index = edge_index.view(2, -1)
        edge_attr = edge_attr.float()
        layer_out = []
        x = F.relu(self.conv1(x, edge_index, edge_attr, batch=batch)).to(opt.device)
        x = self.bn1(x)
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2_0(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[0]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2_1(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[1]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv2_2(x, edge_index, edge_attr, batch=batch))
        x = self.bn1(x)
        x = x + 0.7 * layer_out[2]
        layer_out.append(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_attr, batch=batch))
        x = self.bn2(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.conv4(x, edge_index, edge_attr, batch=batch))
        x = self.bn3(x)
        x = F.dropout(x, p=0.5, training=self.training)

        x = x.view(batch[-1] + 1, -1)

        graph_embedding = self.fc1(x)

        return graph_embedding

"Second_Order_Pooling"
def second_order_pooling(x):
    # 计算二阶关系
    second_order_relations = torch.matmul(x.transpose(1, 0), x)
    # 对二阶关系进行 L2 归一化
    coe_SOP = F.normalize(second_order_relations, p=2, dim=-1)
    # 使用 Softmax 获取权重
    weights = F.softmax(coe_SOP, dim=-2)
    # 对输入进行加权池化
    pooled_output = torch.matmul(x, weights)

    return pooled_output

"Concat Function"
def concat(embedding1, embedding2, embedding3, embedding4):
    # 把graph_embedding、VGG_embedding、second_graph_embedding、second_VGG_embedding进行拼接
    concat_embedding = torch.cat((embedding1, embedding2, embedding3, embedding4), 1)

    return concat_embedding


"Multi-Stream"
class multi_stream(nn.Module):
    def __init__(self, args):
        super(multi_stream, self).__init__()

        self.args = args
        self._setup()
    def _setup(self):
        # C-stream
        self.c_stream_vgg = VGG()
        self.c_stream_matrix = MatrixToFeature()
        # G-stream
        self.g_stream_chebnet_wan = ChebNet_WAN()
        self.g_stream_chebnet_man = ChebNet_MAN()
        self.g_stream_chebnet_lan = ChebNet()
        # 一个全连接层达到二分类的效果
        self.fc = nn.Sequential(
            nn.Linear(800, 512),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(512, 128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 2),
        )
    def forward(self,fc_batch, wml_batch):
    # 使用C-stream得到被试的Embedding
        # c_stream_alex_embedding = self.c_stream_alex(raw_feature_fc)
        c_stream_vgg_embedding = self.c_stream_matrix(fc_batch)
        # 使用G-stream得到被试的Embedding
        # 处理wml_batch
        wan_x, wan_idx, wan_attr = wml_batch.x.to(opt.device), wml_batch.edge_index.to(opt.device), wml_batch.edge_attr.to(opt.device)
        man_x, man_idx, man_attr = wml_batch.man_x.to(opt.device), wml_batch.man_idx.to(opt.device), wml_batch.man_attr.to(opt.device)
        lan_x, lan_idx, lan_attr = wml_batch.lan_x.to(opt.device), wml_batch.lan_idx.to(opt.device), wml_batch.lan_attr.to(opt.device)
        batch = wml_batch.batch.to(opt.device)

        g_stream_chebnet_wan_embedding = self.g_stream_chebnet_wan(wan_x, wan_idx, wan_attr, batch)
        g_stream_chebnet_man_embedding = self.g_stream_chebnet_man(man_x, man_idx, man_attr, batch)
        g_stream_chebnet_lan_embedding = self.g_stream_chebnet_lan(lan_x, lan_idx, lan_attr, batch)
        g_stream_chebnet = torch.cat((g_stream_chebnet_wan_embedding,g_stream_chebnet_man_embedding,g_stream_chebnet_lan_embedding),dim=1).to(opt.device)
# 对C-stream的Embedding进行二阶池化
#         c_stream_alex_embedding_second = second_order_pooling(c_stream_alex_embedding)
        c_stream_vgg_embedding_second = second_order_pooling(c_stream_vgg_embedding)
# 对G-stream的Embedding进行二阶池化
        g_stream_chebnet_second = second_order_pooling(g_stream_chebnet)
        embedding = concat(c_stream_vgg_embedding,g_stream_chebnet,c_stream_vgg_embedding_second,g_stream_chebnet_second)
        pd = self.fc(embedding).to(opt.device)

        return pd

