import numpy as np
import torch.nn
import os
from opt import *
from utils.utils import dataloader, get_node_feature, get_node_label, batch_train, batch_evaluate
from layer import multi_stream, upper_triangle_concat
from tensorboardX import SummaryWriter
from torch.utils.data import Dataset, DataLoader


# 自定义数据集类
class ListDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        # 根据索引返回数据
        return self.data_list[index]


def add_dimensions(tensor):
    # 使用unsqueeze函数增加新的维度
    tensor = tensor.unsqueeze(0)  # 在第0维增加一个新的维度
    tensor = tensor.unsqueeze(0)  # 再在第0维增加一个新的维度
    return tensor


def train():
    print("  Number of training samples %d" % len(train_ind))
    print("  Number of validation samples %d" % len(test_ind))
    print('  Start training...')
    acc = 0
    correct = 0
    best_epo = 0
    for epoch in range(opt.num_iter):
        model.train()
        optimizer.zero_grad()
        with torch.set_grad_enabled(True):
            loss_train, correct_train, acc_train = batch_train(train_fc_loader, train_wml_loader, model, loss_fn, optimizer)
        if opt.log_save:
            writer.add_scalar('train\tloss', loss_train.item(), epoch)
        if opt.log_save:
            writer.add_scalar('train\tacc', acc_train, epoch)

        model.eval()
        with torch.set_grad_enabled(False):
            loss_val, correct_val, acc_val, val_sen, val_spe, val_f1, val_auc = \
                batch_evaluate(test_fc_loader, test_wml_loader, model, loss_fn)

        if opt.log_save:
            writer.add_scalar('val\tloss', loss_val.item(), epoch)
        if opt.log_save:
            writer.add_scalar('val\tacc', acc_val, epoch)

        lr = optimizer.state_dict()['param_groups'][0]['lr']
        if epoch % opt.print_freq == 0:
            print(
                "Epoch: {},\tlr: {:.5f},\ttrain loss: {:.5f},\ttrain acc: {:.5f},\teval loss: {:.5f},\teval acc: {:.5f} ,"
                "\teval spe: {:.5f}\teval_sen: {:.5f}".format(epoch, lr, loss_train.item(), acc_train.item(),
                                                              loss_val.item(), acc_val.item(), val_spe, val_sen))
        if acc_val > acc:
            best_epo = epoch
            acc = acc_val
            correct = correct_val
            aucs[fold] = val_auc
            sens[fold] = val_sen
            spes[fold] = val_spe
            f1[fold] = val_f1
            if (opt.ckpt_path != '') and opt.model_save:
                if not os.path.exists(opt.ckpt_path):
                    os.makedirs(opt.ckpt_path)
                torch.save(model.state_dict(), fold_model_path)
                print("Epoch:{} {} Saved model to:{}".format(epoch, "\u2714", fold_model_path))

    accs[fold] = acc
    corrects[fold] = correct

    print("\r\n => Fold {} , best val_acc {:.5f}, epoch {}".format(fold, acc, best_epo))


def evaluate():
    print("  Number of testing samples %d" % len(test_ind))
    print('  Start testing...')
    model.load_state_dict(torch.load(fold_model_path))
    model.eval()
    _, _, acc_test, test_sen, test_spe, test_f1, test_auc = batch_evaluate(test_fc_loader, test_wml_loader, model, loss_fn)

    accs[fold] = acc_test
    aucs[fold] = test_auc
    sens[fold] = test_sen
    spes[fold] = test_spe
    f1[fold] = test_f1

    print("  Fold {} test accuracy {:.5f}, AUC {:.5f}".format(fold, accs[fold], aucs[fold]))

def save_result():
    with open(f'./result/{opt.dataset}/result.txt', 'a') as f:
        print("========================================================", file=f)
        print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)), file=f)
        print("=> Average test sen in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)), file=f)
        print("=> Average test spe in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)), file=f)
        print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)), file=f)
        print("=> Average test F1-score in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(f1), np.var(f1)), file=f)
        print("========================================================", file=f)


if __name__ == '__main__':
    opt = OptInit().initialize()
    raw_feature_fc, raw_feature_wan, raw_feature_man, raw_feature_lan = get_node_feature(opt.datapath)
    # raw_feature_fc转换为上三角
    raw_feature_fc = upper_triangle_concat(raw_feature_fc)
    # 标签从0开始，否则计算交叉熵损失会报错
    y = get_node_label(opt.datapath) - 1
    labels = torch.tensor(y, dtype=torch.long).to(opt.device)
    dl = dataloader(opt, (raw_feature_fc, raw_feature_wan, raw_feature_man, raw_feature_lan), y)
    # 图分类特征
    wml = (raw_feature_wan, raw_feature_man, raw_feature_lan)

    # k折交叉验证
    n_folds = opt.folds
    cv_splits = dl.data_split(n_folds)
    print(cv_splits)
    corrects = np.zeros(n_folds, dtype=np.int32)
    accs = np.zeros(n_folds, dtype=np.float32)
    sens = np.zeros(n_folds, dtype=np.float32)
    spes = np.zeros(n_folds, dtype=np.float32)
    f1 = np.zeros(n_folds, dtype=np.float32)
    aucs = np.zeros(n_folds, dtype=np.float32)

    for fold in range(n_folds):
        print("\r\n========================== Fold {} ==========================".format(fold))
        train_ind = cv_splits[fold][0]
        test_ind = cv_splits[fold][1]
        # fc训练和测试
        train_fc = []
        test_fc = []
        for i in train_ind:
            train_fc.append(raw_feature_fc[i])
        for i in test_ind:
            test_fc.append(raw_feature_fc[i])
        # 转为dataset形式
        train_fc = ListDataset(train_fc)
        test_fc = ListDataset(test_fc)

        # fc_loader
        train_fc_loader = DataLoader(train_fc, batch_size=opt.batch_size, shuffle=True, num_workers=0)
        test_fc_loader = DataLoader(test_fc, batch_size=opt.batch_size, shuffle=False, num_workers=0)

        # wml_loader
        train_wml_loader = dl.batch_loader(wml, train_ind, y, labels, batch_size=opt.batch_size, shuffle=True)
        test_wml_loader = dl.batch_loader(wml, test_ind, y, labels, batch_size=opt.batch_size, shuffle=False)

        model = multi_stream(opt).to(opt.device)
        print(model)
        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr, weight_decay=opt.wd)
        fold_model_path = opt.ckpt_path + "/fold{}.pth".format(fold)
        if opt.log_save:
            writer = SummaryWriter(f'./{opt.dataset}_log/{fold}')
        if opt.train == 1:
            train()

    print("\r\n========================== Finish ==========================")
    print("=> Average test accuracy in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(accs), np.var(accs)))
    print("=> Average test sensitivity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(sens), np.var(sens)))
    print("=> Average test specificity in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(spes), np.var(spes)))
    print("=> Average test AUC in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(aucs), np.var(aucs)))
    print("=> Average test F1-score in {}-fold CV: {:.5f}({:.4f})".format(n_folds, np.mean(f1), np.var(f1)))
    print("{} Saved model to:{}".format("\u2714", opt.ckpt_path))
    save_result()