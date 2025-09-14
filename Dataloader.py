import os

import numpy as np
import torch
from torch.utils.data import Dataset
from collections import Counter
import pywt


class Load_Dataset(Dataset):
    def __init__(self, dataset, traindata):
        super(Load_Dataset, self).__init__()

        X_train = dataset["samples"]
        y_train = dataset["labels"]

        if traindata:
            print(X_train.shape, y_train.shape, '---------11111--------')
            print(f'This dataset has {max(y_train) + 1} classes')
            unique_classes = torch.unique(y_train) if isinstance(y_train, torch.Tensor) else np.unique(y_train)
            for cls in unique_classes:
                if isinstance(y_train, torch.Tensor):
                    count = (y_train == cls).sum().item()
                    cls_val = int(cls.item())
                else:
                    count = (y_train == cls).sum()
                    cls_val = int(cls)
                print(f"Class {cls_val}: {count} samples")


        if isinstance(X_train, np.ndarray):
            self.x_data = torch.from_numpy(X_train).float()
            self.y_data = torch.from_numpy(y_train).long()
        else:
            self.x_data = X_train.float()
            self.y_data = y_train.long()

        self.len = X_train.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len


def wavelet_transform(x, weak=True):
    """小波变换函数"""

    if len(x.shape) == 3:  # (batch, seq_len, features)
        b, n, f = x.shape
        x_transformed = np.zeros_like(x)

        for i in range(b):
            for j in range(f):
                signal = x[i, :, j]
                b_g, b_d = pywt.dwt(signal, 'db2')

                if weak:
                    b_d = b_d + np.random.random(b_d.shape) * 0.1
                else:
                    b_g = b_g + np.random.random(b_g.shape) * 0.1


                reconstructed = pywt.idwt(b_g, b_d, 'db2')


                if len(reconstructed) > n:
                    reconstructed = reconstructed[:n]
                elif len(reconstructed) < n:
                    reconstructed = np.pad(reconstructed, (0, n - len(reconstructed)), mode='constant')

                x_transformed[i, :, j] = reconstructed

        return x_transformed
    else:

        b_g, b_d = pywt.dwt(x, 'db2')

        if weak:
            b_d = b_d + np.random.random(b_d.shape) * 0.1
        else:
            b_g = b_g + np.random.random(b_g.shape) * 0.1

        a_ = pywt.idwt(b_g, b_d, 'db2')
        return a_


def data_generator(args):
    """数据生成器，返回DataLoader"""
    train_dataset = torch.load(os.path.join(args.data_path, "train_LT" + str(args.IR) + ".pt"))
    val_dataset = torch.load(os.path.join(args.data_path, "val.pt"))
    test_dataset = torch.load(os.path.join(args.data_path, "test.pt"))

    train_dataset = Load_Dataset(train_dataset, True)
    val_dataset = Load_Dataset(val_dataset, False)
    test_dataset = Load_Dataset(test_dataset, False)


    num_workers = 4 if torch.cuda.is_available() else 0
    pin_memory = torch.cuda.is_available()

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=args.batchsize,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=args.batchsize,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader, val_loader, test_loader