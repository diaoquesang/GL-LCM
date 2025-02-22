import cv2 as cv
import lpips
import numpy as np
import os
from config import config
from openpyxl import Workbook

import torch
from skimage.metrics import structural_similarity as ssim
from math import log10, sqrt
import torchvision.transforms as transforms

def cal_BSR(cxr_path, gt_path, bs_path):
    cxr = cv.imread(cxr_path, 0)
    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)

    cxr = cxr / 255
    gt = gt / 255
    bs = bs / 255

    bone = cv.subtract(cxr, gt)


    gt = cv.resize(gt, (config.image_size, config.image_size))
    bs = cv.resize(bs, (config.image_size, config.image_size))
    bone = cv.resize(bone, (config.image_size, config.image_size))

    bs += np.average(cv.subtract(gt, bs))

    bias = cv.subtract(gt, bs)
    bias[bias < 0] = 0

    BSR = 1 - np.sum(bias ** 2) / np.sum(bone ** 2)
    return BSR


def cal_MSE(gt_path, bs_path):
    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)



    gt = cv.resize(gt, (config.image_size, config.image_size))
    bs = cv.resize(bs, (config.image_size, config.image_size))

    MSE = np.mean((gt - bs) ** 2)
    MSE = 2*lpips.l2(gt,bs)
    return MSE


def cal_SSIM(gt_path, bs_path):
    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)

    gt = gt / 255
    bs = bs / 255



    gt = cv.resize(gt, (config.image_size, config.image_size))
    bs = cv.resize(bs, (config.image_size, config.image_size))

    SSIM = ssim(gt, bs, channel_axis=None, data_range=1)

    return SSIM


def cal_PSNR(gt_path, bs_path):
    mse = cal_MSE(gt_path,bs_path)
    max_pixel = 1

    PSNR = 20 * log10(max_pixel / sqrt(mse))
    return PSNR




def cal_LPIPS(gt_path, bs_path):
    lplps_model = lpips.LPIPS()

    gt = cv.imread(gt_path, 0)
    bs = cv.imread(bs_path, 0)


    gt = cv.resize(gt, (config.image_size, config.image_size))
    bs = cv.resize(bs, (config.image_size, config.image_size))

    gt = transforms.ToTensor()(gt)
    bs = transforms.ToTensor()(bs)

    gt = torch.unsqueeze(gt, dim=0)
    bs = torch.unsqueeze(bs, dim=0)

    LPIPS = lplps_model(gt, bs).item()

    return LPIPS


if __name__ == "__main__":
    # 创建一个新的工作簿
    wb = Workbook()

    # 选择默认的工作表
    ws = wb.active

    CXR_path = "SZCH-X-Rays/CXR"
    GT_path = "SZCH-X-Rays/BS"
    BS_path = "lcm_output_bs/Fusion_BS"

    BSR_list = []
    MSE_list = []
    SSIM_list = []
    PSNR_list = []
    LPIPS_list = []
    HBD_list = []
    ws.append(["Filename", "BSR", "MSE", "SSIM", "PSNR", "LPIPS"])
    txt = 'SZCH_testset.txt'
    with open(txt, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    file_names = [line.strip() for line in lines]

    for filename in os.listdir(BS_path):
        if filename in file_names:
            pass
        else:
            continue
        cxr_path = os.path.join(CXR_path, filename)
        gt_path = os.path.join(GT_path, filename)
        bs_path = os.path.join(BS_path, filename)

        BSR = cal_BSR(cxr_path, gt_path, bs_path)
        MSE = cal_MSE(gt_path, bs_path)
        SSIM = cal_SSIM(gt_path, bs_path)
        PSNR = cal_PSNR(gt_path, bs_path)
        LPIPS = cal_LPIPS(gt_path, bs_path)

        BSR_list.append(BSR)
        MSE_list.append(MSE)
        SSIM_list.append(SSIM)
        PSNR_list.append(PSNR)
        LPIPS_list.append(LPIPS)
        print(f"{filename} BSR: {BSR} MSE: {MSE}  SSIM:{SSIM} PSNR:{PSNR} LPIPS:{LPIPS}")
        # 可以使用append方法插入一行数据
        ws.append([filename, BSR, MSE, SSIM, PSNR, LPIPS])

    ws.append(["Mean",
               np.mean(np.array(BSR_list)),
               np.mean(np.array(MSE_list)),
               np.mean(np.array(SSIM_list)),
               np.mean(np.array(PSNR_list)),
               np.mean(np.array(LPIPS_list))])
    ws.append(["Std",
               np.std(np.array(BSR_list)),
               np.std(np.array(MSE_list)),
               np.std(np.array(SSIM_list)),
               np.std(np.array(PSNR_list)),
               np.std(np.array(LPIPS_list))])
    print("Average BSR:", np.mean(np.array(BSR_list)), "Std:", np.std(np.array(BSR_list)))
    print("Average MSE:", np.mean(np.array(MSE_list)), "Std:", np.std(np.array(MSE_list)))
    print("Average SSIM:", np.mean(np.array(SSIM_list)), "Std:", np.std(np.array(SSIM_list)))
    print("Average PSNR:", np.mean(np.array(PSNR_list)), "Std:", np.std(np.array(PSNR_list)))
    print("Average LPIPS:", np.mean(np.array(LPIPS_list)), "Std:", np.std(np.array(LPIPS_list)))

    # 保存工作簿到文件
    wb.save("sample.xlsx")
