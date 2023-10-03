import os
import random
import sys

import cv2
import lmdb
import numpy as np
import torch
import torch.utils.data as data

import data.util as util

class Val_nii_dataset(data.Dataset):

    def __init__(self, opt):
        super(Val_nii_dataset).__init__()
        self.opt = opt
        self.env = None
        self.n_frame = self.opt["N_frames"]

        GT_rootpath = self.opt['dataroot_GT']
        GT_list = sorted(os.listdir(GT_rootpath))
        self.GT_paths = [os.path.join(GT_rootpath, i) for i in GT_list]
        len_val = len(GT_list)

        LQ_rootpath = self.opt['dataroot_LQ']
        LQ_list = sorted(os.listdir(LQ_rootpath))
        self.LQ_paths = [os.path.join(LQ_rootpath, i) for i in LQ_list]
        # print(self.LQ_paths)

    def __getitem__(self, index):
        """
        采样次数：(len(self.GT_paths)-1)//(self.n_frame-1)
        """
        #将采样index映射到val_loader
        index = index*(self.n_frame-1)
        image_GT_list = []
        image_LQ_list = []
        for i in range(self.n_frame):
            GT_path = self.GT_paths[index+i]
            img_GT = util.read_img(self.env, GT_path)
            image_GT_list.append(img_GT)

            LQ_path = self.LQ_paths[index+i]
            img_LQ = util.read_img(self.env, LQ_path)
            image_LQ_list.append(img_LQ)

        img_LQ = np.stack(image_LQ_list, axis=0)
        img_GT = np.stack(image_GT_list, axis=0)
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (0, 3, 1, 2)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (0, 3, 1, 2)))).float()

        time_list = []
        for i in range(5):
            time_list.append(torch.Tensor([i / (5 - 1)]))
        time_Tensors = torch.cat(time_list)
        if self.opt['N_frames']==5:
            time_tensor = time_Tensors[[1, 2, 3]]
        elif self.opt['N_frames']==4:
            time_tensor = time_Tensors[[1, 3]]
        elif self.opt['N_frames']==3:
            time_tensor = time_Tensors[[2]]
        elif self.opt['N_frames']==7:
            time_tensor = torch.tensor([1/6, 2/6, 3/6, 4/6, 5/6])
        
        if self.opt['use_time'] == True:
            return {'LQs': img_LQ[[0, -1], :, :, :], 'GT': img_GT[1:-1, :, :, :], 'time': time_tensor}
        else:
            return {'LQs': img_LQ[[0, -1], :, :, :], 'GT': img_GT[1:-1, :, :, :]}
        
    ## 控制index的次数
    def __len__(self):
        return (len(self.GT_paths)-1)//(self.n_frame-1)