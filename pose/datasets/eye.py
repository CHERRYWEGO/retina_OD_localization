from __future__ import print_function, absolute_import
# import matplotlib
# import matplotlib.pyplot as plt
# matplotlib.use('Qt5Agg')

import os
import numpy as np
import json
import random
import math

import re
import torch
import torch.utils.data as data
from PIL import Image
from scipy.misc import imresize
from torchvision import transforms

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
import pandas as pd
import os
import cv2

# box_label
def eye_data(imageDir, labelDir):
    raw_labels1 = pd.read_csv(labelDir, header=None)
    train = []

    for file in os.listdir(imageDir):
        imagename = file

        if raw_labels1.loc[raw_labels1.iloc[:, 0] == int(imagename.split(".")[0])].shape[0] != 1:
            continue
        raw_label1 = raw_labels1.loc[raw_labels1.iloc[:, 0] == int(imagename.split(".")[0])].values[0]

        # if raw_labels1.loc[int(imagename.split(".")[0]) - 1].values.shape[0] != 1:
        #     continue
        # raw_label1 = raw_labels1.loc[int(imagename.split(".")[0]) - 1].values

        # label_x = (raw_label1[1] + raw_label1[3])/2
        # label_y = (raw_label1[2] + raw_label1[4])/2
        # labels = [label_x, label_y]
        # labels = [raw_label1[1], raw_label1[2]]

        train.append([file, np.array(raw_label1[1:5])])

    df = pd.DataFrame(train, columns=['file', 'label'])
    return df

# # fovea and od
# def eye_data(imageDir, labelDir):
#     raw_labels1 = pd.read_csv(labelDir, header=None)
#     train = []
#
#     for file in os.listdir(imageDir):
#         imagename = file
#
#         if raw_labels1.loc[raw_labels1.iloc[:, 0] == int(imagename.split(".")[0])].shape[0] != 1:
#             continue
#         raw_label1 = raw_labels1.loc[raw_labels1.iloc[:, 0] == int(imagename.split(".")[0])].values[0]
#
#         if(raw_label1[5] < 0):
#             continue
#
#         # if raw_labels1.loc[int(imagename.split(".")[0]) - 1].values.shape[0] != 1:
#         #     continue
#         # raw_label1 = raw_labels1.loc[int(imagename.split(".")[0]) - 1].values
#
#         # label_x = (raw_label1[1] + raw_label1[3])/2
#         # label_y = (raw_label1[2] + raw_label1[4])/2
#         # labels = [label_x, label_y]
#         # labels = [raw_label1[1], raw_label1[2]]
#
#         train.append([file, np.array(raw_label1[1:5]), np.array(raw_label1[5:7])])
#
#     df = pd.DataFrame(train, columns=['file', 'od_label', 'fovea_label'])
#     return df

# def eye_data2(imageDir):
#     # raw_labels1 = pd.read_csv(labelDir)
#     train = []
#
#     # img_root = imageDir
#     # imageDir = os.path.join(img_root, kind)
#     # patterns = re.split(r"\\|/", imageDir)
#     # prefix = patterns[-2].split("_")[0] + "/" + patterns[-1].split("_")[0] + "/"
#     # label_mask = kind_masks[patterns[-1].split("_")[0]]
#     for file in os.listdir(imageDir):
#         train.append([file])
#     df = pd.DataFrame(train, columns=['file'])
#     return df

# class Eye1(data.Dataset):
#     def __init__(self, df, img_folder, mask_folder, inp_res=256, out_res=64, sigma=1,
#                  scale_factor=0.25, rot_factor=30, label_type='Gaussian', nparts=2, test_condition=False):
#         self.df = df
#         self.img_folder = img_folder  # root image folders
#         self.mask_folder = mask_folder
#         self.inp_res = inp_res
#         self.out_res = out_res
#         self.sigma = sigma
#         self.scale_factor = scale_factor
#         self.rot_factor = rot_factor
#         self.label_type = label_type
#         self.nparts = nparts
#         self.test_condition = test_condition
#
#         # create train/val split
#         # with open(jsonfile) as anno_file:
#         #     self.anno = json.load(anno_file)
#         #
#         # self.train, self.valid = [], []
#         # for idx, val in enumerate(self.anno):
#         #     if val['isValidation'] == True:
#         #         self.valid.append(idx)
#         #     else:
#         #         self.train.append(idx)
#         self.mean, self.std = self._compute_mean()
#
#     def _compute_mean(self):
#         # meanstd_file = '/home/shiluj/workspace/optic disc/pytorchpose/data/Eye/mean.pth.tar'
#         meanstd_file = './data/Eye/mean.pth.tar'
#         if isfile(meanstd_file):
#             meanstd = torch.load(meanstd_file)
#         else:
#             mean = torch.zeros(3)
#             std = torch.zeros(3)
#             for i in range(0, len(self.df)):
#                 # a = self.anno[index]
#                 # img_path = os.path.join(self.img_folder, self.df.iloc[i, 2])
#                 img_path = os.path.join(self.img_folder, self.df.iloc[i, 0])
#                 img, ratio = self._load_image(img_path)  # CxHxW
#                 img = transforms.ToTensor()(img)
#                 mean += img.view(img.size(0), -1).mean(1)
#                 std += img.view(img.size(0), -1).std(1)
#             mean /= len(self.df)
#             std /= len(self.df)
#             meanstd = {
#                 'mean': mean,
#                 'std': std,
#             }
#             torch.save(meanstd, meanstd_file)
#             # if self.is_train:
#             print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
#             print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))
#
#         return meanstd['mean'], meanstd['std']
#     def _load_image(self, img_path):
#         img = Image.open(img_path).convert('RGB')
#         ratio = self.inp_res / max(img.height, img.width)
#         img = transforms.Resize(int(self.inp_res * min(img.height, img.width) / max(img.height, img.width)))(img)
#         if img.width != self.inp_res or img.height != self.inp_res:
#             img = transforms.Pad(padding=(0, 0, self.inp_res - img.width, self.inp_res - img.height))(img)
#         return img, ratio
#
#     def __getitem__(self, index):
#         img_path = os.path.join(self.img_folder, self.df.iloc[index, 0])
#
#         # Adjust center/scale slightly to avoid cropping limbs
#
#         # For single-person pose estimation with a centered/scaled figure
#         # nparts = 2
#         img, ratio = self._load_image(img_path)
#         # plt.imshow(img)
#         # ori_img = img
#         # img.save("example.jpg")
#         img = transforms.ToTensor()(img)
#         ratio = ratio * self.out_res / self.inp_res
#
#         # Color
#         if not self.test_condition:
#             img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
#             img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
#             img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
#
#         # Prepare image and groundtruth map
#         inp = color_normalize(img, self.mean, self.std)
#
#         # Generate ground truth
#         # pts = torch.Tensor(self.df.iloc[index, 1])
#         pts_box = self.df.iloc[index, 1]
#         label_x = (pts_box[0] + pts_box[2])/2
#         label_y = (pts_box[1] + pts_box[3])/2
#         # label_x = pts_box[0]
#         # label_y = pts_box[1]
#         pts_fovea = self.df.iloc[index, 2]
#         pts = torch.Tensor([[label_x, label_y], [pts_fovea[0], pts_fovea[1]]])
#
#
#         # target = torch.zeros(self.nparts, self.out_res, self.out_res)
#         target = np.zeros((self.nparts, self.out_res, self.out_res), np.float32)
#         npts = (pts.clone() * ratio).long()
#         npts_box = pts_box * ratio
#
#         # lside = int(round((npts_box[2] - npts_box[0]) / 2))
#         # sside = int(round((npts_box[3] - npts_box[1]) / 2))
#         #
#         # for i in range(self.nparts):
#         #     cv2.ellipse(target[i], (npts[0], npts[1]), (lside, sside), 0, 0, 360, (1, 1, 1), -1)
#
#         for i in range(self.nparts):
#             if npts[i, 0] > 0:
#                 target[i] = draw_labelmap(target[i], npts[i], self.sigma, type=self.label_type)
#
#         # Meta info
#         meta = {'index': index, 'img_path': img_path, 'pts_box': pts_box,
#                 'pts': pts, 'npts': npts, "ratio": torch.Tensor([ratio])}
#
#         return inp, target, meta
#
#     def __len__(self):
#         return len(self.df)
#
class Eye2(data.Dataset):
    def __init__(self, df, img_folder, mask_folder, inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', nparts=1, test_condition=False):
        self.df = df
        self.img_folder = img_folder  # root image folders
        self.mask_folder = mask_folder
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.nparts = nparts
        self.test_condition = test_condition

        # create train/val split
        # with open(jsonfile) as anno_file:
        #     self.anno = json.load(anno_file)
        #
        # self.train, self.valid = [], []
        # for idx, val in enumerate(self.anno):
        #     if val['isValidation'] == True:
        #         self.valid.append(idx)
        #     else:
        #         self.train.append(idx)
        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
        # meanstd_file = '/home/shiluj/workspace/optic disc/pytorchpose/data/Eye/mean.pth.tar'
        meanstd_file = './data/Eye/mean.pth.tar'
        if isfile(meanstd_file):
            meanstd = torch.load(meanstd_file)
        else:
            mean = torch.zeros(3)
            std = torch.zeros(3)
            for i in range(0, len(self.df)):
                # a = self.anno[index]
                # img_path = os.path.join(self.img_folder, self.df.iloc[i, 2])
                img_path = os.path.join(self.img_folder, self.df.iloc[i, 0])
                img, ratio = self._load_image(img_path)  # CxHxW
                img = transforms.ToTensor()(img)
                mean += img.view(img.size(0), -1).mean(1)
                std += img.view(img.size(0), -1).std(1)
            mean /= len(self.df)
            std /= len(self.df)
            meanstd = {
                'mean': mean,
                'std': std,
            }
            torch.save(meanstd, meanstd_file)
            # if self.is_train:
            print('    Mean: %.4f, %.4f, %.4f' % (meanstd['mean'][0], meanstd['mean'][1], meanstd['mean'][2]))
            print('    Std:  %.4f, %.4f, %.4f' % (meanstd['std'][0], meanstd['std'][1], meanstd['std'][2]))

        return meanstd['mean'], meanstd['std']
    def _load_image(self, img_path):
        img = Image.open(img_path).convert('RGB')
        ratio = self.inp_res / max(img.height, img.width)
        img = transforms.Resize(int(self.inp_res * min(img.height, img.width) / max(img.height, img.width)))(img)
        if img.width != self.inp_res or img.height != self.inp_res:
            img = transforms.Pad(padding=(0, 0, self.inp_res - img.width, self.inp_res - img.height))(img)
        return img, ratio

    def __getitem__(self, index):
        # if self.is_train:
        #     a = self.anno[self.train[index]]
        # else:
        #     a = self.anno[self.valid[index]]
        # img_name = self.df.iloc[index, 0]
        # img_folder = img_name + "/" + self.df.iloc[index, 1]
        # img_f = os.path.join(self.img_folder, self.df.iloc[index, 0])

        img_path = os.path.join(self.img_folder, self.df.iloc[index, 0])
        # pts = torch.Tensor(self.df.iloc[index, 1])

        pts_box = self.df.iloc[index, 1]
        label_x = (pts_box[0] + pts_box[2])/2
        label_y = (pts_box[1] + pts_box[3])/2
        # label_x = pts_box[0]
        # label_y = pts_box[1]
        pts = torch.Tensor([label_x, label_y])

        # print(img_path)
        # pts[:, 0:2] -= 1  # Convert pts to zero based
        # '/home/shiluj/workspace/C.Localization/1. Original Images/b. Testing Set/IDRiD_085.jpg'

        # Adjust center/scale slightly to avoid cropping limbs

        # For single-person pose estimation with a centered/scaled figure
        # nparts = 2
        img, ratio = self._load_image(img_path)
        # plt.imshow(img)
        ori_img = img
        img.save("example.jpg")
        img = transforms.ToTensor()(img)
        ratio = ratio * self.out_res / self.inp_res
        # img = im_to_torch(imresize(scipy.misc.imread(img_path, mode='RGB'), (self.inp_res, self.inp_res)))  # CxHxW
        # img = Image.open(img_path).convert('RGB')
        # img = transforms.Resize(int(self.inp_res * min(img.height, img.width) / max(img.height, img.width)))(img)
        # if img.width != self.inp_res or img.height != self.inp_res:
        #     img = transforms.Pad(padding=(0, 0, self.inp_res - img.width, self.inp_res - img.height))(img)
        # img = transforms.Resize((self.inp_res, self.inp_res))(img)

        # ratio = self.inp_res / img.shape[1]
        # if self.is_train:
        #     s = s * torch.randn(1).mul_(sf).add_(1).clamp(1 - sf, 1 + sf)[0]
        #     r = torch.randn(1).mul_(rf).clamp(-2 * rf, 2 * rf)[0] if random.random() <= 0.6 else 0

            # Flip
            # if random.random() <= 0.5:
            #     img = torch.from_numpy(fliplr(img.numpy())).float()
            #     pts = shufflelr(pts, width=img.size(2), dataset='mpii')
            #     c[0] = img.size(2) - c[0]

        # Color
        if not self.test_condition:
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = color_normalize(img, self.mean, self.std)

        # Generate ground truth
        # tpts = pts.clone()
        # target = torch.zeros(nparts, self.out_res, self.out_res)
        target = torch.zeros(self.nparts, self.out_res, self.out_res)
        cover = []
        # kind_mask = self.kind_masks[self.df.iloc[index, 2]]
        npts = (pts.clone() * ratio).long()
        for i in range(self.nparts):
            # if tpts[i, 2] > 0: # This is evil!!
            # cover.append(tpts[i][2])
            if npts[0] > 0:
                # tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
                target[i] = draw_labelmap(target[i], npts, self.sigma, type=self.label_type)

        # Meta info
        meta = {'index': index, 'img_path': img_path,
                'pts_box': pts_box,
                'pts': pts, 'npts': npts, "ratio": torch.Tensor([ratio])}

        return inp, target, meta

    # def __getitem__(self, index):
    #     img_path = os.path.join(self.img_folder, self.df.iloc[index, 0])
    #     mask_path = os.path.join(self.mask_folder, self.df.iloc[index, 0])
    #     # pts = torch.Tensor(self.df.iloc[index, 1])
    #
    #     img, ratio = self._load_image(img_path)
    #     # plt.imshow(img)
    #     ori_img = img
    #     img.save("example.jpg")
    #     img = transforms.ToTensor()(img)
    #     ratio = ratio * self.out_res / self.inp_res
    #
    #     # Color
    #     img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
    #     img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
    #     img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
    #
    #     # Prepare image and groundtruth map
    #     inp = color_normalize(img, self.mean, self.std)
    #
    #     # Generate ground truth
    #     # tpts = pts.clone()
    #     # target = torch.zeros(nparts, self.out_res, self.out_res)
    #     # target = torch.zeros(self.nparts, self.out_res, self.out_res)
    #     cover = []
    #     # kind_mask = self.kind_masks[self.df.iloc[index, 2]]
    #     # npts = (pts.clone() * ratio).long()
    #     # for i in range(self.nparts):
    #     #     # if tpts[i, 2] > 0: # This is evil!!
    #     #     # cover.append(tpts[i][2])
    #     #     if npts[0] > 0:
    #     #         # tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
    #     #         target[i] = draw_labelmap(target[i], npts, self.sigma, type=self.label_type)
    #
    #     # Meta info
    #     meta = {'index': index, 'img_path': img_path, 'mask_path': mask_path, "ratio": torch.Tensor([ratio])}
    #
    #     return inp, meta

    def __len__(self):
        return len(self.df)
