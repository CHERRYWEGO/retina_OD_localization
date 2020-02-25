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
# from scipy.misc import imresize
from torchvision import transforms

from pose.utils.osutils import *
from pose.utils.imutils import *
from pose.utils.transforms import *
import pandas as pd
import os
import cv2

def eye_testdata(imageDir):
    train = []
    for file in os.listdir(imageDir):
        train.append([file])
    df = pd.DataFrame(train, columns=['file'])
    return df

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

class Eye(data.Dataset):
    def __init__(self, df, img_folder,
            # mask_folder, 
            inp_res=256, out_res=64, sigma=1,
            scale_factor=0.25, rot_factor=30, label_type='Gaussian', nparts=1, test_condition=False):
        self.df = df
        self.img_folder = img_folder  # root image folders
        # self.mask_folder = mask_folder
        self.inp_res = inp_res
        self.out_res = out_res
        self.sigma = sigma
        self.scale_factor = scale_factor
        self.rot_factor = rot_factor
        self.label_type = label_type
        self.nparts = nparts
        self.test_condition = test_condition

        self.mean, self.std = self._compute_mean()

    def _compute_mean(self):
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
        img_path = os.path.join(self.img_folder, self.df.iloc[index, 0])
        mask_path = os.path.join(self.mask_folder, self.df.iloc[index, 0])

        img, ratio = self._load_image(img_path)
        img = transforms.ToTensor()(img)
        ratio = ratio * self.out_res / self.inp_res

        # Color
        if not self.test_condition:
            img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
            img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)

        # Prepare image and groundtruth map
        inp = color_normalize(img, self.mean, self.std)

        # Generate ground truth
        # pts = torch.Tensor(self.df.iloc[index, 1])
        pts_box = self.df.iloc[index, 1]
        label_x = (pts_box[0] + pts_box[2])/2
        label_y = (pts_box[1] + pts_box[3])/2
        # label_x = pts_box[0]
        # label_y = pts_box[1]
        pts = torch.Tensor([label_x, label_y])

        target = np.zeros((self.nparts, self.out_res, self.out_res), np.float32)
        npts = (pts.clone() * ratio).long()

        for i in range(self.nparts):
            if npts[0] > 0:
                target[i] = draw_labelmap(target[i], npts, self.sigma, type=self.label_type)

        # Meta info
        meta = {'index': index, 'img_path': img_path, 'mask_path': mask_path,
                'pts_box': pts_box, "ratio": torch.Tensor([ratio])}

        return inp, target, meta

    def __len__(self):
        return len(self.df)
#
class Eyetest(data.Dataset):
    def __init__(self, df, img_folder,
                # mask_folder, 
                inp_res=256, out_res=64, sigma=1,
                 scale_factor=0.25, rot_factor=30, label_type='Gaussian', nparts=1, test_condition=False):
        self.df = df
        self.img_folder = img_folder  # root image folders
        # self.mask_folder = mask_folder
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
        img_path = os.path.join(self.img_folder, self.df.iloc[index, 0])
        # mask_path = os.path.join(self.mask_folder, self.df.iloc[index, 0])
        # pts = torch.Tensor(self.df.iloc[index, 1])
    
        img, ratio = self._load_image(img_path)
        # plt.imshow(img)
        # ori_img = img
        # img.save("example.jpg")
        img = transforms.ToTensor()(img)
        ratio = ratio * self.out_res / self.inp_res
    
        # # Color
        # img[0, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        # img[1, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
        # img[2, :, :].mul_(random.uniform(0.8, 1.2)).clamp_(0, 1)
    
        # Prepare image and groundtruth map
        inp = color_normalize(img, self.mean, self.std)
    
        # Generate ground truth
        # tpts = pts.clone()
        # target = torch.zeros(nparts, self.out_res, self.out_res)
        # target = torch.zeros(self.nparts, self.out_res, self.out_res)
        # cover = []
        # kind_mask = self.kind_masks[self.df.iloc[index, 2]]
        # npts = (pts.clone() * ratio).long()
        # for i in range(self.nparts):
        #     # if tpts[i, 2] > 0: # This is evil!!
        #     # cover.append(tpts[i][2])
        #     if npts[0] > 0:
        #         # tpts[i, 0:2] = to_torch(transform(tpts[i, 0:2] + 1, c, s, [self.out_res, self.out_res], rot=r))
        #         target[i] = draw_labelmap(target[i], npts, self.sigma, type=self.label_type)
    
        # Meta info
        meta = {'index': index, 'img_path': img_path, "ratio": torch.Tensor([ratio])}
    
        return inp, meta

    def __len__(self):
        return len(self.df)
