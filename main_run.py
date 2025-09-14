import argparse
import time
import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, matthews_corrcoef, f1_score
from torch import optim
import os
from sklearn.cluster import KMeans

from Dataloader import data_generator, wavelet_transform
from models import *
from torch.utils.data import Dataset
import math

parser = argparse.ArgumentParser()

parser.add_argument('--epoches', default=100, type=int)
parser.add_argument('--data_path', default='data/HAR', type=str)
parser.add_argument('--seq_len', default=128, type=int)
parser.add_argument('--indim', default=9, type=int)
parser.add_argument('--classes', default=6, type=int)
parser.add_argument('--lr', default=0.005, type=float)

parser.add_argument('--IR', default=100, type=int)

parser.add_argument('--down_sampling_layers', default=2, type=int)
parser.add_argument('--batchsize', default=128, type=int)
parser.add_argument('--channel', default=128, type=int)
parser.add_argument('--temporal_size', default=128, type=int)
parser.add_argument('--feat_size', default=32, type=int)

# ===== 损失函数类型选择参数 =====
parser.add_argument('--loss_type', default='mse', type=str, choices=['mse', 'kl_js'],
                    help='Choose loss type: mse (our method) or kl_js')





class EarlyStopping:
    def __init__(self, patience=15, verbose=False, dump=False, args=None, device='cpu'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_ls = None
        self.device = device
        if args is not None:
            self.best_model = Model(args).to(device)
        else:
            self.best_model = None
        self.early_stop = False
        self.dump = dump
        self.val_loss_min = np.inf
        self.delta = 0
        self.trace_func = print

    def __call__(self, val_loss, model, epoch):
        ls = val_loss
        if self.best_ls is None:
            self.best_ls = ls
        elif ls > self.best_ls + self.delta:
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience} with {epoch}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_ls = ls
            self.counter = 0




def kl_divergence(s_out, t_out, tau=1.0):
    # s_out, t_out: [B, C] logits
    # Step 1: softmax with temperature
    p = F.softmax(t_out / tau, dim=1)  # teacher
    q = F.log_softmax(s_out / tau, dim=1)  # student (log-softmax for KL)

    # Step 2: KL(P || Q) = sum(P * (logP - logQ))
    loss = F.kl_div(q, p, reduction="batchmean") * (tau ** 2)
    return loss



def js_divergence(p_logits, q_logits, tau=1.0):
    """计算Jensen-Shannon散度"""
    p = F.softmax(p_logits / tau, dim=1)
    q = F.softmax(q_logits / tau, dim=1)


    m = (p + q) / 2.0


    kl_pm = F.kl_div(F.log_softmax(p_logits / tau, dim=1), m, reduction="batchmean")
    kl_qm = F.kl_div(F.log_softmax(q_logits / tau, dim=1), m, reduction="batchmean")


    js_div = 0.5 * (kl_pm + kl_qm) * (tau ** 2)
    return js_div


# =================================


def hellinger_distance(s_out, t_out, tau=1.0):
    p = F.softmax(t_out / tau, dim=1)
    q = F.softmax(s_out / tau, dim=1)

    # Step 2: compute Hellinger distance
    sqrts = torch.sqrt(p) - torch.sqrt(q)  # sqrt(P) - sqrt(Q)
    h = torch.norm(sqrts, p=2, dim=1) / torch.sqrt(torch.tensor(2.0))  # batch-wise

    # Step 3: mean over batch
    loss = h.mean()
    return loss


def Test(test_loader, bestmodel, final, device):
    predicts = None
    labels = None
    bestmodel.eval()

    bestmodel = bestmodel.to(device)

    for data, label in test_loader:
        data, label = data.to(device), label.to(device)
        out = bestmodel(data)
        out = torch.argmax(out, dim=-1).reshape(-1)
        label = label.reshape(-1)

        if predicts is None:
            predicts = out.detach().cpu()
            labels = label.detach().cpu()
        else:
            predicts = torch.cat([predicts, out.detach().cpu()])
            labels = torch.cat([labels, label.detach().cpu()])

    X = None
    if final == True:
        num_classes = torch.max(labels).item() + 1
        correct = torch.zeros(num_classes)
        total = torch.zeros(num_classes)

        for i in range(num_classes):
            mask = (labels == i)
            total[i] = mask.sum()
            correct[i] = (predicts[mask] == labels[mask]).sum()

        per_class_acc = correct / total.clamp(min=1)
        X = []
        for i in range(num_classes):
            X.append(per_class_acc[i].item() * 100)
            print(f"Class {i} Accuracy: {per_class_acc[i].item() * 100:.2f}%", '总数是:', int(total[i].item()))

    acc = accuracy_score(predicts, labels) * 100
    f1 = f1_score(predicts, labels, average="macro") * 100
    mcc = matthews_corrcoef(predicts, labels)
    printtext = "ACC:{:.2f}".format(acc) + '% ' + 'F1:{:.2f}'.format(
        f1) + '% ' + 'MCC:{:.4f}'.format(
        mcc)
    print(printtext)
    return acc, f1, mcc, X


def Trainer(args, smodel, s_optimizer, tmodel, t_optimizer,
            train_loader, val_loader, test_loader, early_stopping,
            s_scheduler, t_scheduler, device):

    PredLossFun = nn.CrossEntropyLoss()

    for epoch in range(1, args.epoches + 1):
        train_losses, sval_losses, tval_losses, test_losses = [], [], [], []
        smodel.train()
        tmodel.train()

        for data, label in train_loader:
            data, label = data.to(device), label.to(device)

            s_optimizer.zero_grad()
            t_optimizer.zero_grad()

            data_weak = torch.from_numpy(
                wavelet_transform(data.cpu().numpy(), weak=True)
            ).float().to(device)
            data_strong = torch.from_numpy(
                wavelet_transform(data.cpu().numpy(), weak=False)
            ).float().to(device)

            sout = smodel(data)
            sout_weak = smodel(data_weak)
            sout_strong = smodel(data_strong)

            tout = tmodel(data)
            tout_weak = tmodel(data_weak)
            tout_strong = tmodel(data_strong)


            if args.loss_type == 'mse':
                # MSE
                sloss = (F.cross_entropy(sout, label) + F.cross_entropy(sout_weak, label) + F.cross_entropy(sout_strong,
                                                                                                            label)) + (
                                F.mse_loss(sout, tout_weak) + F.mse_loss(sout_weak,
                                                                         tout_strong))

                tloss = (F.cross_entropy(tout, label) + F.cross_entropy(tout_weak, label) + F.cross_entropy(tout_strong,
                                                                                                            label)) + (
                                F.mse_loss(tout, sout_weak) + F.mse_loss(tout_weak, sout_strong))
            else:  # args.loss_type == 'kl_js'


                s_class_loss = F.cross_entropy(sout, label) + F.cross_entropy(sout_weak, label) + F.cross_entropy(
                    sout_strong, label)
                t_class_loss = F.cross_entropy(tout, label) + F.cross_entropy(tout_weak, label) + F.cross_entropy(
                    tout_strong, label)


                s_rd_loss = kl_divergence(sout, tout_weak) + kl_divergence(sout_weak, tout_strong)
                t_rd_loss = kl_divergence(tout, sout_weak) + kl_divergence(tout_weak, sout_strong)


                alignment_loss = js_divergence(sout, tout)

                sloss = s_class_loss + s_rd_loss + 0.5 * alignment_loss
                tloss = t_class_loss + t_rd_loss + 0.5 * alignment_loss
            # =======================================

            train_losses.append(sloss.item())
            sloss.backward(retain_graph=True)
            tloss.backward()

            s_optimizer.step()
            t_optimizer.step()

        with torch.no_grad():
            smodel.eval()
            tmodel.eval()
            for data, label in val_loader:
                data, label = data.to(device), label.to(device)
                out = smodel(data)
                loss = PredLossFun(out, label)
                sval_losses.append(loss.item())
                out = tmodel(data)
                loss = PredLossFun(out, label)
                tval_losses.append(loss.item())

        with torch.no_grad():
            smodel.eval()
            for data, label in test_loader:
                data, label = data.to(device), label.to(device)
                out = smodel(data)
                loss = PredLossFun(out, label)
                test_losses.append(loss.item())

        print('epoch:{0:}, train_loss:{1:.5f}, val_loss:{2:.5f}, test_loss:{3:.5f}'.format(epoch,
                                                                                           np.mean(train_losses),
                                                                                           np.mean(sval_losses),
                                                                                           np.mean(test_losses)))

        s_scheduler.step(np.mean(sval_losses))
        t_scheduler.step(np.mean(tval_losses))

        early_stopping(np.mean(sval_losses), smodel, epoch)
        if early_stopping.early_stop:
            print("Early stopping with best_ls:{}".format(early_stopping.best_ls))
            break
        if np.isnan(np.mean(sval_losses)) or np.isnan(np.mean(train_losses)):
            break

        Test(test_loader, smodel, False, device)

def main(args, i):
    SEED = i
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = data_generator(args)
    smodel = Model(args).float().to(DEVICE)
    tmodel = Model(args).float().to(DEVICE)

    s_optimizer = optim.Adam(smodel.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)
    t_optimizer = optim.Adam(tmodel.parameters(), lr=args.lr, betas=(.9, .99), weight_decay=5e-4)

    s_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        s_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    t_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        t_optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
    )
    early_stopping = EarlyStopping(patience=10, verbose=True, dump=False)

    Trainer(args, smodel, s_optimizer, tmodel, t_optimizer,
            train_loader, val_loader, test_loader,
            early_stopping, s_scheduler, t_scheduler, DEVICE)

    acc, f1, mcc, X = Test(test_loader,
                           smodel,
                           True, DEVICE)
    return acc, f1, mcc, X


if __name__ == '__main__':
    args = parser.parse_args()
    for IR in [100, 50]:
        args.IR = IR
        start = time.time()
        ACC = []
        F1 = []
        MCC = []
        X = []
        for i in range(2):
            acc, f1, mcc, x = main(args, i)
            print(acc, f1, mcc)
            ACC.append(acc)
            F1.append(f1)
            MCC.append(mcc)
            X.append(x)
            torch.cuda.empty_cache()

        X = np.stack(X, axis=0)
        print(X)
        print(ACC)
        print(F1)
        print(MCC)
        print('\033[31m===================', args.IR, '===============================\033[0m')
        accmean = np.mean(ACC)
        f1mean = np.mean(F1)
        mccmean = np.mean(MCC)
        print('\033[31m ACC:', accmean, '\033[0m')
        print('\033[31m F1:', f1mean, '\033[0m')
        print('\033[31m MCC:', mccmean, '\033[0m')
        print('\033[31m', np.mean(X, axis=0), '\033[0m')
        print('\033[31m', np.std(X, axis=0), '\033[0m')
        print(time.time() - start, '秒-------------')
        print('\033[31m====================================================\033[0m')
        print('\033[31m====================================================\033[0m')
        print('\033[31m====================================================\033[0m')
        print('\033[31m====================================================\033[0m')