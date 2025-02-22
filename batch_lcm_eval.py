from config import config
import numpy as np

from dataset import myC2BDataset
from transform import myTransform
from torch.utils.data import DataLoader
from diffusers import LCMScheduler

from tqdm import tqdm
import cv2 as cv
import torch
import time
import os

from monai.utils import set_determinism

set_determinism(42)


def eval():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置运行环境
    output_path = os.path.join("lcm_output_bs", "BS")
    masked_output_path = os.path.join("lcm_output_bs", "Masked_BS")
    fusion_output_path = os.path.join("lcm_output_bs", "Fusion_BS")

    cxr_path = os.path.join("SZCH-X-Rays", "CXR")
    masked_cxr_path = os.path.join("SZCH-X-Rays", "Masked_CXR")
    mask_path = os.path.join("SZCH-X-Rays", "Mask")

    model = torch.load("masked_lcm-600-2024-12-19-myModel.pth").to(device).eval()
    VQGAN = torch.load("2024-12-12-Mask-SZCH-VQGAN.pth").to(device).eval()
    testset_list = "SZCH.txt"
    myTestSet = myC2BDataset(testset_list, cxr_path, masked_cxr_path, myTransform['testTransform'])
    myTestLoader = DataLoader(myTestSet, batch_size=1, shuffle=False)
    # 设置噪声调度器
    noise_scheduler = LCMScheduler(num_train_timesteps=config.num_train_timesteps,
                                   clip_sample=config.clip_sample,
                                   clip_sample_range=config.initial_clip_sample_range_g)
    noise_scheduler.set_timesteps(config.num_infer_timesteps)
    with torch.no_grad():
        progress_bar = tqdm(enumerate(myTestLoader), total=len(myTestLoader), ncols=100)
        total_start = time.time()
        for step, batch in progress_bar:
            cxr = batch[0].to(device=device, non_blocking=True).float()
            masked_cxr = batch[1].to(device=device, non_blocking=True).float()
            filename = batch[2][0]
            cxr_copy = np.array(cxr.detach().to("cpu"))
            cxr_copy = np.squeeze(cxr_copy)  # HW
            cxr_copy = cxr_copy * 0.5 + 0.5
            cxr_copy *= 255
            cxr_copy = cxr_copy.astype(np.int8)

            cxr = VQGAN.encode_stage_2_inputs(cxr)
            masked_cxr = VQGAN.encode_stage_2_inputs(masked_cxr)

            noise = torch.randn_like(cxr).to(device)
            sample = torch.cat((noise, cxr), dim=1).to(device)  # BCHW
            masked_sample = torch.cat((noise, masked_cxr), dim=1).to(device)  # BCHW

            for j, t in tqdm(enumerate(noise_scheduler.timesteps)):
                residual = model(sample, torch.Tensor((t,)).to(device).long()).to(device)
                masked_residual = model(masked_sample, torch.Tensor((t,)).to(device).long()).to(device)
                # masked_residual = (1 - config.alpha) * residual + config.alpha * masked_residual
                masked_residual = config.alpha * masked_residual + (1 - config.alpha) * torch.randn_like(
                    masked_residual).to(device) / torch.std(masked_residual)

                noise_scheduler = LCMScheduler(num_train_timesteps=config.num_train_timesteps,
                                               clip_sample=config.clip_sample,
                                               clip_sample_range=
                                               config.initial_clip_sample_range_g
                                               + config.clip_rate * j
                                               )
                noise_scheduler.set_timesteps(config.num_infer_timesteps)
                sample = noise_scheduler.step(residual, t, sample).prev_sample

                noise_scheduler = LCMScheduler(num_train_timesteps=config.num_train_timesteps,
                                               clip_sample=config.clip_sample,
                                               clip_sample_range=
                                               config.initial_clip_sample_range_l
                                               + config.clip_rate * j
                                               )
                noise_scheduler.set_timesteps(config.num_infer_timesteps)
                masked_sample = noise_scheduler.step(masked_residual, t, masked_sample).prev_sample

                sample = torch.cat((sample[:, :4], cxr), dim=1)  # BCHW
                masked_sample = torch.cat((masked_sample[:, :4], masked_cxr), dim=1).to(device)  # BCHW
                if config.output_feature_map:
                    bs_show = np.array(sample[:, 0].detach().to("cpu"))
                    bs_show = np.squeeze(bs_show)  # HW
                    bs_show = bs_show * 0.5 + 0.5
                    bs_show = np.clip(bs_show, 0, 1)

                    masked_bs_show = np.array(masked_sample[:, 0].detach().to("cpu"))
                    masked_bs_show = np.squeeze(masked_bs_show)  # HW
                    masked_bs_show = masked_bs_show * 0.5 + 0.5
                    masked_bs_show = np.clip(masked_bs_show, 0, 1)

                    if not config.use_server:
                        cv.imshow("win1", bs_show)
                        cv.imshow("win2", masked_bs_show)
                        cv.waitKey(1)

            mask = cv.imread(os.path.join(mask_path, filename), 0)
            mask[mask < 255] = 0

            bs = VQGAN.decode((sample[:, :4]))
            bs = np.array(bs.detach().to("cpu"))
            bs = np.squeeze(bs)  # HW
            bs = bs * 0.5 + 0.5
            bs[cxr_copy == 0] = 0

            masked_bs = VQGAN.decode((masked_sample[:, :4]))
            masked_bs = np.array(masked_bs.detach().to("cpu"))
            masked_bs = np.squeeze(masked_bs)  # HW
            masked_bs = masked_bs * 0.5 + 0.5
            masked_bs[mask > 0] = masked_bs[mask > 0] + np.mean(bs[mask > 0]) - np.mean(masked_bs[mask > 0])
            masked_bs[cxr_copy == 0] = 0
            if not config.use_server:
                cv.imshow("win3", bs)
                cv.imshow("win4", masked_bs)
                cv.waitKey(1)

            bs *= 255
            cv.imwrite(os.path.join(output_path, filename), bs)
            masked_bs *= 255
            cv.imwrite(os.path.join(masked_output_path, filename), masked_bs)


            num_labels, labels, stats, _ = cv.connectedComponentsWithStats(mask)
            min_area = 100
            for i in range(1, num_labels):
                if stats[i, cv.CC_STAT_AREA] < min_area:
                    labels[labels == i] = 0
            mask[labels == 0] = 0

            br = cv.boundingRect(mask)
            p = (br[0] + br[2] // 2, br[1] + br[3] // 2)

            masked_bs = np.clip(masked_bs, 0, 255)
            masked_bs = cv.cvtColor(masked_bs, cv.COLOR_GRAY2BGR).astype(np.uint8)
            bs = np.clip(bs, 0, 255)
            bs = cv.cvtColor(bs, cv.COLOR_GRAY2BGR).astype(np.uint8)

            fusion_bs = cv.seamlessClone(masked_bs, bs, mask, p, cv.MONOCHROME_TRANSFER)
            # cv.rectangle(fusion_bs, br, (0, 255, 0), 2)
            # fusion_bs[mask==255]=(255, 0, 0)

            cv.imwrite(os.path.join(fusion_output_path, filename), fusion_bs)

        total_time = time.time() - total_start
        print(f"Total time: {total_time}.")


if __name__ == "__main__":
    eval()
