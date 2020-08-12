# !coding:utf-8

import os
import json
import gzip

import torch
import torch.utils.data as data
import torchvision

import matplotlib.pyplot as plt

class MnistTransformed(object):
    def __init__(self, img):
        self.img = img

    def __call__(self):
        print("trans for img")

class MnistDataset(data.Dataset):
    def __init__(self, dataset_path, mode='train'):
        if not dataset_path:
            print("dataset path cannot be null")
            return

        data = json.load(gzip.open(dataset_path))
        train, val, test = data

        if mode == 'train':
            self.imgs, self.labels = train[0], train[1]
        elif mode == 'valid':
            self.imgs, self.labels = val[0], val[1]
        elif mode == 'test':
            self.imgs, self.labels = test[0], test[1]
        else:
            raise Exception("model can only be one of ['train', 'val', 'eval']")
        
        print(type(self.imgs[0]), type(self.labels[0]))
        print("训练集数据量：", len(self.imgs), len(self.labels))
        print(len(self.imgs[0]))

    def __getitem__(self, index):
        img, label = torch.tensor(self.imgs[index], dtype=torch.float32).view(28, 28), torch.tensor(self.labels[index], dtype=torch.float32)
        return img, label

    def __len__(self):
        return len(self.imgs)

if __name__ == "__main__":
    print(torch.cuda.is_available())
    dataset = MnistDataset("E:\Workspace\ML\dataset\mnist\mnist.json.gz")

    train_loader = data.DataLoader(dataset=dataset, batch_size=10, shuffle=True)

    train_iter = iter(train_loader)
    img,lbl = train_iter.next()

    print(img.size(), lbl.size())
    print(img, lbl)

    # plt.imshow(img.numpy())
