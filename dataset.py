from torch.utils.data import Dataset
import pandas as pd
import cv2 as cv
import os
from config import config
import torch
import numpy as np


class mySingleDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, img_dir, transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.img_dir = img_dir  # 读取图像路径
        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        img_path = self.img_dir  # 读取图片文件夹路径

        file = self.filelist.iloc[idx, 0]  # 读取文件名
        image = cv.imread(os.path.join(img_path, file))  # 用openCV的imread函数读取图像

        if self.transform:
            image = self.transform(image)  # 图像预处理
        return image, file  # 返回图像和名称


class myDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, cxr_dir, bs_dir,
                 transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.cxr_dir = cxr_dir  # 读取图像路径
        self.bs_dir = bs_dir  # 读取图像路径

        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        file = self.filelist.iloc[idx, 0]  # 读取文件名
        cxr = cv.imread(os.path.join(self.cxr_dir, file))  # 用openCV的imread函数读取图像
        bs = cv.imread(os.path.join(self.bs_dir, file))  # 用openCV的imread函数读取图像

        if self.transform:
            cxr = self.transform(cxr)  # 图像预处理
            bs = self.transform(bs)  # 图像预处理

        return cxr, bs, file  # 返回图像和标签

class myC2BDataset(Dataset):  # 定义数据集类
    def __init__(self, filelist, cxr_dir, masked_cxr_dir,
                 transform=None):  # 传入参数(标签路径,图像路径,图像预处理方式,标签预处理方式)
        self.cxr_dir = cxr_dir  # 读取图像路径
        self.masked_cxr_dir = masked_cxr_dir  # 读取图像路径

        self.transform = transform  # 读取图像预处理方式
        self.filelist = pd.read_csv(filelist, sep="\t", header=None)  # 读取文件名列表

    def __len__(self):
        return len(self.filelist)  # 读取文件名数量作为数据集长度

    def __getitem__(self, idx):  # 从数据集中取出数据
        file = self.filelist.iloc[idx, 0]  # 读取文件名
        cxr = cv.imread(os.path.join(self.cxr_dir, file))  # 用openCV的imread函数读取图像
        masked_cxr = cv.imread(os.path.join(self.masked_cxr_dir, file))  # 用openCV的imread函数读取图像

        if self.transform:
            cxr = self.transform(cxr)  # 图像预处理
            masked_cxr = self.transform(masked_cxr)  # 图像预处理

        return cxr, masked_cxr, file  # 返回图像和标签