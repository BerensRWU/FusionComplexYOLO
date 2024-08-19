"""
# -*- coding: utf-8 -*-
-----------------------------------------------------------------------------------
# Author: Nguyen Mau Dung
# DoC: 2020.07.05
# email: nguyenmaudung93.kstn@gmail.com
-----------------------------------------------------------------------------------
# Description: This script for the KITTI dataset

# Refer: https://github.com/ghimiredhikura/Complex-YOLOv3
"""

import sys
import os
import random
import itertools
import numpy as np

from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import cv2

from LiDAR_fog_sim.fog_simulation import simulate_fog_wrapper

sys.path.append('../')

from data_process_astyx import transformation, astyx_bev_utils, astyx_data_utils
import config.kitti_config as cnf


class astyxDataset(Dataset):
    def __init__(self, dataset_dir, mode='train', lidar_transforms=None, 
                 aug_transforms=None, multiscale=False, num_samples=546, mosaic=False,
                 random_padding=False, configs=None):

        self.dataset_dir = dataset_dir
        self.saved_fn = configs.saved_fn
        assert mode in ['train', 'valid', 'test'], f'Invalid mode: {mode}'
        self.mode = mode
        self.is_test = (self.mode == 'test')

        self.dist_prop_lidar = configs.dist_prop_lidar
        self.dist_prop_radar = configs.dist_prop_radar
        self.dist_types_lidar = configs.disturb_types_training_lidar
        self.dist_levels_lidar = configs.disturb_levels_training_lidar
        self.dist_types_radar = configs.disturb_types_training_radar
        self.dist_levels_radar = configs.disturb_levels_training_radar

        self.disturb_lidar_data = self.dist_types_lidar != None
        self.disturb_radar_data = self.dist_types_radar != None

        self.low_fusion = configs.low_fusion
        self.high_fusion = configs.high_fusion
        self.radar = configs.radar
        self.lidar = configs.lidar
        self.VR = configs.VR
        self.multiscale = multiscale
        self.lidar_transforms = lidar_transforms
        self.aug_transforms = aug_transforms
        self.img_size = cnf.BEV_WIDTH
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0
        self.mosaic = mosaic
        self.random_padding = random_padding
        self.mosaic_border = [-self.img_size // 2, -self.img_size // 2]
        
        self.lidar_dir = os.path.join(self.dataset_dir, 'lidar_vlp16')
        self.radar_dir = os.path.join(self.dataset_dir, 'radar_6455')
        self.image_dir = os.path.join(self.dataset_dir, 'camera_front')
        self.calib_dir = os.path.join(self.dataset_dir, 'calibration')
        self.label_dir = os.path.join(self.dataset_dir, 'groundtruth_obj3d')

        rng = np.random.default_rng(seed=configs.set_seed)

        #rng.shuffle(split)
        
        if self.mode == "train":
            #self.sample_id_list = split[:int(np.ceil(num_samples*0.8))]
            split_txt_path = os.path.join(self.dataset_dir, 'split', 'train.txt')
        else:
            #self.sample_id_list = split[int(np.ceil(num_samples*0.8)):]
            split_txt_path = os.path.join(self.dataset_dir, 'split', 'valid.txt')
        #self.sample_id_list = [*list(range(39)),*list(range(130,546))]
        
        self.image_idx_list = [x.strip() for x in open(split_txt_path).readlines()]
        
        if self.is_test:
            self.sample_id_list = [int(sample_id) for sample_id in self.image_idx_list]
        else:
            self.sample_id_list = self.remove_invalid_idx(self.image_idx_list)
        
        self.num_samples = len(self.sample_id_list)

    def __getitem__(self, index):
        if False:
            return self.load_img_only(index)
        else:
            if self.mosaic:
                img_files, rgb_map, targets = self.load_mosaic(index)

                return img_files[0], rgb_map, targets
            else:
                return self.load_img_with_targets(index)

    def load_img_only(self, index):
        """Load only image for the testing phase"""
        pass
        """sample_id = int(self.sample_id_list[index])
        
        calib = self.get_calib(sample_id)
        
        if self.low_fusion:
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
            pcData = np.concatenate([pcData,self.get_radar(sample_id)])
                
        elif self.high_fusion:
            pcData = self.get_radar(sample_id)
            b = astyx_bev_utils.removePoints(pcData, cnf.boundary)
            rgb_map_radar = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
            b = astyx_bev_utils.removePoints(pcData, cnf.boundary)
            rgb_map_lidar = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)

            img_file = os.path.join(self.image_dir, '{:06d}.jpg'.format(sample_id))

            return img_file, rgb_map_radar, rgb_map_lidar

        elif self.radar:
            pcData = self.get_radar(sample_id)
        elif self.lidar:
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
        else:
            raise NotImplementedError
            
        b = astyx_bev_utils.removePoints(pcData, cnf.boundary)
        rgb_map = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
        img_file = os.path.join(self.image_dir, '{:06d}.jpg'.format(sample_id))

        return img_file, rgb_map"""

    def load_img_with_targets(self, index):
        """Load images and targets for the training and validation phase"""
        sample_id = int(self.sample_id_list[index])

        #lidarData = self.get_lidar(sample_id)
        objects = self.get_label(sample_id)
        calib = self.get_calib(sample_id)
        
        labels, noObjectLabels = astyx_bev_utils.read_labels_for_bevbox(objects)

        target = astyx_bev_utils.build_yolo_target(labels)
        img_file = os.path.join(self.image_dir, '{:06d}.jpg'.format(sample_id))

        # on image space: targets are formatted as (box_idx, class, x, y, w, l, im, re)
        n_target = len(target)
        targets = torch.zeros((n_target, 8))
        if n_target > 0:
            targets[:, 1:] = torch.from_numpy(target)
        if self.mode == "train":
            dist_prop_lidar = self.dist_prop_lidar
            dist_prop_radar = self.dist_prop_radar
        else:
            dist_prop_lidar = 1.
            dist_prop_radar = 1.

        if self.low_fusion:
            radarData = self.get_radar(sample_id)
            lidarData = self.get_lidar(sample_id)
            intensity = lidarData[:,3].reshape(-1,1)
            lidarData = calib.lidar2ref(lidarData[:,0:3])
            lidarData = np.concatenate([lidarData,intensity],1)
            radarData = astyx_bev_utils.removePoints(radarData, cnf.boundary)
            lidarData = astyx_bev_utils.removePoints(lidarData, cnf.boundary)
            if self.disturb_lidar_data:
                if np.random.uniform(low=0.0, high=1.0) < dist_prop_lidar:
                    lidarData = self.disturb_pc(pcData=lidarData, types=self.dist_types_lidar, levels=self.dist_levels_lidar, sample_id=sample_id, calib=calib, radar = False)
            if self.disturb_radar_data:
                if np.random.uniform(low=0.0, high=1.0) < dist_prop_radar:
                    radarData = self.disturb_pc(pcData=radarData, types=self.dist_types_radar, levels=self.dist_levels_radar, sample_id=sample_id, calib=calib, radar = True)

            pcData = np.concatenate([lidarData,radarData])
            
        elif self.high_fusion:
            pcData = self.get_radar(sample_id)
            pcData = astyx_bev_utils.removePoints(pcData, cnf.boundary)
            
            if self.disturb_radar_data:
                if np.random.uniform() < dist_prop_radar:
                    pcData = self.disturb_pc(pcData=pcData, types=self.dist_types_radar, levels=self.dist_levels_radar, sample_id=sample_id, calib=calib, radar = True)
            
            
            if self.lidar_transforms is not None:
                pcData, labels[:, 1:] = self.lidar_transforms(pcData, labels[:, 1:])
            
            b = astyx_bev_utils.removePoints(pcData, cnf.boundary, True)

            rgb_map_radar = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            rgb_map_radar = torch.from_numpy(rgb_map_radar).float()

            
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
            pcData = astyx_bev_utils.removePoints(pcData, cnf.boundary)
            if self.disturb_lidar_data:
                if np.random.uniform() < dist_prop_lidar:
                    pcData = self.disturb_pc(pcData=pcData, types=self.dist_types_lidar, levels=self.dist_levels_lidar, sample_id=sample_id, calib=calib, radar = False)
                    
            if self.lidar_transforms is not None:
                pcData, labels[:, 1:] = self.lidar_transforms(pcData, labels[:, 1:])
                
            b = astyx_bev_utils.removePoints(pcData, cnf.boundary, True)

            rgb_map_lidar = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
            rgb_map_lidar = torch.from_numpy(rgb_map_lidar).float()

                    

            return img_file, rgb_map_radar, rgb_map_lidar, targets

                
        elif self.radar:
            pcData = self.get_radar(sample_id)
            pcData = astyx_bev_utils.removePoints(pcData, cnf.boundary)
            
            if self.disturb_radar_data:
                if np.random.uniform() < dist_prop_radar:
                    pcData = self.disturb_pc(pcData=pcData, types=self.dist_types_radar, levels=self.dist_levels_radar, sample_id=sample_id, calib=calib, radar = True)

        elif self.lidar:
            pcData = self.get_lidar(sample_id)
            intensity = pcData[:,3].reshape(-1,1)
            pcData = calib.lidar2ref(pcData[:,0:3])
            pcData = np.concatenate([pcData,intensity],1)
            pcData = astyx_bev_utils.removePoints(pcData, cnf.boundary)
            if self.disturb_lidar_data:
                if np.random.uniform() < dist_prop_lidar:
                    pcData = self.disturb_pc(pcData=pcData, types=self.dist_types_lidar, levels=self.dist_levels_lidar, sample_id=sample_id, calib=calib, radar = False)
        else:
            raise NotImplementedError
            
        if self.lidar_transforms is not None:
            pcData, labels[:, 1:] = self.lidar_transforms(pcData, labels[:, 1:])
            
        b = astyx_bev_utils.removePoints(pcData, cnf.boundary, True)

        rgb_map = astyx_bev_utils.makeBVFeature(b, cnf.DISCRETIZATION, cnf.boundary)
        rgb_map = torch.from_numpy(rgb_map).float()

        if self.aug_transforms is not None:
            rgb_map, targets = self.aug_transforms(rgb_map, targets)
        return img_file, rgb_map, targets

    def disturb_pc(self, pcData, types, levels, sample_id, calib,radar=False):
        for type, level in zip(types,levels):
            if type == "snow":
                pcData = self.add_snow(sample_id, level, calib)
            elif type == "fog":
                pcData = self.add_fog(pcData, level)
            elif type == "random_noise":
                pcData = self.add_random_noise(pcData, level)
            elif type == "random_loss":
                pcData = self.lose_random_points(pcData,level)
            elif type == "random_shift":
                pcData = self.add_random_shift(pcData, level)
            elif type == "blobs":
                if radar:
                    pcData = self.add_blobs_radar(pcData, level)
                else:
                    pcData = self.add_blobs_lidar(pcData, level)
            elif type == "intensity":
                pcData = self.add_intensity_noise(pcData, level)
            elif type == "None" or type == None:
                pcData = pcData
            else:
                raise NotImplementedError(type)
        return pcData

    def add_intensity_noise(self, pcData, level):
        if pcData.shape[1] == 4:
            rand_i = np.random.uniform(-100 * level, 100 * level, pcData.shape[0])
            pcData[:,3] += rand_i
        else:
            rand_m = np.random.uniform(-70 * level, 70 * level, pcData.shape[0])
            rand_v = np.random.uniform(-5 * level, 5 * level, pcData.shape[0])
            pcData[:,3] += rand_m
            pcData[:,4] += rand_v
        return pcData
    
    def add_snow(self,idx, level, calib):
        level = int(level*4)
        rep = np.random.randint(1,5)

        pcData = np.loadtxt(f'{self.dataset_dir}/../src/LiDAR_snow_sim/snow_data/rand{rep}/level_{level}/lidar_vlp16/{idx:06d}.txt', dtype=np.float32)
        intensity = pcData[:, 3].reshape(-1, 1)
        pcData = calib.lidar2ref(pcData[:, 0:3])
        pcData = np.concatenate([pcData, intensity], 1)
        return pcData

    def add_random_noise(self, pcData, level):
        rand_x = np.random.uniform(cnf.boundary["minX"], cnf.boundary["maxX"], int(level*pcData.shape[0]))
        rand_y = np.random.uniform(cnf.boundary["minY"], cnf.boundary["maxY"], int(level*pcData.shape[0]))
        rand_z = np.random.uniform(cnf.boundary["minZ"], cnf.boundary["maxZ"], int(level*pcData.shape[0]))
        if pcData.shape[1] == 4:
            rand_i = np.random.uniform(0,100, int(level*pcData.shape[0]))
            rand_points = np.array([rand_x, rand_y, rand_z, rand_i]).T
        else:
            rand_m = np.random.uniform(0, 70, int(level * pcData.shape[0]))
            rand_v = np.random.uniform(-5, 5, int(level * pcData.shape[0]))
            rand_points = np.array([rand_x, rand_y, rand_z, rand_m, rand_v]).T
        pcData = np.concatenate((pcData,rand_points), axis = 0)
        return pcData
        
    def add_random_shift(self, pcData, level):
        rand_shift_factor = np.random.uniform(-2 * level, 2 * level, pcData.shape[0])
        rand_shift_factor = np.random.uniform(-2 * level, 2 * level, pcData.shape[0])
        pcData[:,0:3] = pcData[:,0:3] + pcData[:,0:3] * (rand_shift_factor / np.linalg.norm(pcData[:,0:3], axis = 1)).reshape(-1,1)
        
        return pcData

    def add_blobs_lidar(self, pcData, level):
        amount_blobs = np.random.randint(np.max([1,16 * level**2 - 2]), 16 * level**2 + 3) if level != 0 else 0
        count = 0
        while count < amount_blobs:
            sphere = []
            offset = [np.random.uniform(cnf.boundary["minX"], cnf.boundary["maxX"]),
                      np.random.uniform(cnf.boundary["minY"], cnf.boundary["maxY"]),
                      np.random.uniform(cnf.boundary["minZ"], cnf.boundary["maxZ"])]
            dist = np.linalg.norm(offset)
            if dist < 7:
                continue
            count += 1
            theta_length = np.max([1, int(np.round(16 - 3/10*dist) - 3*offset[2])])
            phi_length = np.max([1,int(np.round(50 - dist))])

            r = np.random.uniform(0.1 + level**2*2,1 + level**2*2)

            if offset[2] < -1:
                theta_list = np.linspace(0, np.pi-1, num = theta_length)
            elif offset[2] > 1:
                theta_list = np.linspace(1, np.pi, num = theta_length)
            else:
                theta_list = np.linspace(1, np.pi-1, num = theta_length)

            phi_list = np.linspace(0.5, np.pi-0.5, num = phi_length)

            for theta, phi in itertools.product(theta_list,phi_list):
                x = r * np.sin(theta) * np.cos(phi) + np.random.uniform(-0.1,0.1)
                y = -r * np.sin(theta) * np.sin(phi) + np.random.uniform(-0.1,0.1)
                z = r * np.cos(theta) + np.random.uniform(-0.1,0.1)
                sphere += [[x,y,z]]
            sphere = np.array(sphere)
            sphere = sphere[sphere[:,2] > cnf.boundary["minZ"]+0.2]
            
            phi = -np.arctan2(offset[0],offset[1])

            theta = -np.arcsin(offset[2]/dist)
            sphere = rotate_points(sphere,rt_matrix(phi,theta))

            sphere += offset

            rand_i = np.random.uniform(0,100)+ np.random.normal(0,4, (len(sphere),1))
            rand_blob = np.concatenate((sphere, rand_i),axis=1)

            pcData = np.concatenate((pcData,rand_blob), axis = 0)
        return pcData
        
    def add_blobs_radar(self, pcData, level):
        amount_blobs = np.random.randint(np.max([1,32 * level**2 - 2]), 32 * level**2 + 2)
        for i in range(int(amount_blobs)):

            size_blob = np.random.uniform(np.max([1,4 * level-1]),4 * level+1)
            points_blob = np.random.randint(np.max([1,100 * level**4 - 0]), 100 * level**4+20)
            
            rand_x = np.random.normal(scale = size_blob/2, size= points_blob) + np.random.uniform(cnf.boundary["minX"], cnf.boundary["maxX"]) 
            rand_y = np.random.normal(scale = size_blob/2, size= points_blob) + np.random.uniform(cnf.boundary["minY"], cnf.boundary["maxY"])
            rand_z = np.random.uniform(0, 2, points_blob) + np.random.uniform(cnf.boundary["minZ"], cnf.boundary["maxZ"])
            
            rand_m = np.random.uniform(0, 70, points_blob)
            rand_v = np.random.uniform(-5, 5, points_blob)
            
            rand_blob = np.array([rand_x, rand_y, rand_z, rand_m]).T
            pcData = np.concatenate((pcData,rand_blob), axis = 0)
            
        return pcData
        
    def lose_random_points(self, pcData, level):
        keep = np.random.choice(pcData.shape[0],size=int(pcData.shape[0]*(1-level)), replace=False)
        return pcData[keep]

    def add_fog(self, pcData, level):
        level = int(level * 4)
        return simulate_fog_wrapper(pcData, level)

    def load_mosaic(self, index):
        """loads images in a mosaic
        Refer: https://github.com/ultralytics/yolov5/blob/master/utils/datasets.py
        """

        targets_s4 = []
        img_file_s4 = []
        if self.random_padding:
            yc, xc = [int(random.uniform(-x, 2 * self.img_size + x)) for x in self.mosaic_border]  # mosaic center
        else:
            yc, xc = [self.img_size, self.img_size]  # mosaic center

        indices = [index] + [random.randint(0, self.num_samples - 1) for _ in range(3)]  # 3 additional image indices
        for i, index in enumerate(indices):
            img_file, img, targets = self.load_img_with_targets(index)
            img_file_s4.append(img_file)

            c, h, w = img.size()  # (3, 608, 608), torch tensor

            # place img in img4
            if i == 0:  # top left
                img_s4 = torch.full((c, self.img_size * 2, self.img_size * 2), 0.5, dtype=torch.float)
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, self.img_size * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, self.img_size * 2), min(self.img_size * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

            img_s4[:, y1a:y2a, x1a:x2a] = img[:, y1b:y2b, x1b:x2b]  # img_s4[ymin:ymax, xmin:xmax]
            padw = x1a - x1b
            padh = y1a - y1b

            # on image space: targets are formatted as (box_idx, class, x, y, w, l, sin(yaw), cos(yaw))
            if targets.size(0) > 0:
                targets[:, 2] = (targets[:, 2] * w + padw) / (2 * self.img_size)
                targets[:, 3] = (targets[:, 3] * h + padh) / (2 * self.img_size)
                targets[:, 4] = targets[:, 4] * w / (2 * self.img_size)
                targets[:, 5] = targets[:, 5] * h / (2 * self.img_size)

            targets_s4.append(targets)
        if len(targets_s4) > 0:
            targets_s4 = torch.cat(targets_s4, 0)
            torch.clamp(targets_s4[:, 2:4], min=0., max=(1. - 0.5 / self.img_size), out=targets_s4[:, 2:4])

        return img_file_s4, img_s4, targets_s4

    def __len__(self):
        return len(self.sample_id_list)

    def remove_invalid_idx(self, image_idx_list):
        """Discard samples which don't have current training class objects, which will not be used for training."""

        sample_id_list = []
        for sample_id in image_idx_list:
            sample_id = int(sample_id)
            objects = self.get_label(sample_id)
            calib = self.get_calib(sample_id)
            labels, noObjectLabels = astyx_bev_utils.read_labels_for_bevbox(objects)
            #if not noObjectLabels:
            #    labels[:, 1:] = transformation.camera_to_lidar_box(labels[:, 1:], calib.V2C, calib.R0,
            #                                                       calib.P)  # convert rect cam to velo cord

            valid_list = []
            for i in range(labels.shape[0]):
                if int(labels[i, 0]) in cnf.CLASS_NAME_TO_ID.values():
                    if self.check_point_cloud_range(labels[i, 1:4]):
                        valid_list.append(labels[i, 0])

            if len(valid_list) > 0:
                sample_id_list.append(sample_id)

        return sample_id_list

    def check_point_cloud_range(self, xyz):
        """
        :param xyz: [x, y, z]
        :return:
        """
        x_range = [cnf.boundary["minX"], cnf.boundary["maxX"]]
        y_range = [cnf.boundary["minY"], cnf.boundary["maxY"]]
        z_range = [cnf.boundary["minZ"], cnf.boundary["maxZ"]]

        if (x_range[0] <= xyz[0] <= x_range[1]) and (y_range[0] <= xyz[1] <= y_range[1]) and \
                (z_range[0] <= xyz[2] <= z_range[1]):
            return True
        return False

    def collate_fn(self, batch):
        if self.high_fusion:
            paths, imgs_radar, imgs_lidar, targets = list(zip(*batch))
            # Remove empty placeholder targets
            targets = [boxes for boxes in targets if boxes is not None]
            # Add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            targets = torch.cat(targets, 0)
            # Selects new image size every tenth batch
            if (self.batch_count % 10 == 0) and self.multiscale and (not self.mosaic):
                self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            # Resize images to input shape
            imgs_radar = torch.stack(imgs_radar)
            if self.img_size != cnf.BEV_WIDTH:
                imgs_radar = F.interpolate(imgs_radar, size=self.img_size, mode="bilinear", align_corners=True)
            # Resize images to input shape
            imgs_lidar = torch.stack(imgs_lidar)
            if self.img_size != cnf.BEV_WIDTH:
                imgs_lidar = F.interpolate(imgs_lidar, size=self.img_size, mode="bilinear", align_corners=True)
            self.batch_count += 1

            return paths, imgs_radar, imgs_lidar, targets
        else:
            paths, imgs, targets = list(zip(*batch))
            # Remove empty placeholder targets
            targets = [boxes for boxes in targets if boxes is not None]
            # Add sample index to targets
            for i, boxes in enumerate(targets):
                boxes[:, 0] = i
            targets = torch.cat(targets, 0)
            # Selects new image size every tenth batch
            if (self.batch_count % 10 == 0) and self.multiscale and (not self.mosaic):
                self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
            # Resize images to input shape
            imgs = torch.stack(imgs)
            if self.img_size != cnf.BEV_WIDTH:
                imgs = F.interpolate(imgs, size=self.img_size, mode="bilinear", align_corners=True)
            self.batch_count += 1

            return paths, imgs, targets

    def get_image(self, idx):
        img_file = os.path.join(self.image_dir, f'{idx:06d}.jpg')
        # assert os.path.isfile(img_file)
        return cv2.imread(img_file)  # (H, W, C) -> (H, W, 3) OpenCV reads in BGR mode

    def get_lidar(self, idx):
        lidar_file = os.path.join(self.lidar_dir, f'{idx:06d}.txt')
        assert os.path.isfile(lidar_file)
        lidar = np.loadtxt(lidar_file, dtype=np.float32, skiprows = 1)
        lidar = lidar[:,0:4]

        return lidar
        
    def get_radar(self, idx):
        radar_file = os.path.join(self.radar_dir, f'{idx:06d}.txt')
        assert os.path.isfile(radar_file)
        radar = np.loadtxt(radar_file, dtype=np.float32, skiprows = 2)
        if self.VR:
            radar = radar[:,[0,1,2,3]]
        else:
            radar = radar[:,[0,1,2,4]]

        return radar

    def get_calib(self, idx):
        calib_file = os.path.join(self.calib_dir, f'{idx:06d}.json')
        assert os.path.isfile(calib_file)
        return astyx_data_utils.Calibration(calib_file)

    def get_label(self, idx):
        label_file = os.path.join(self.label_dir, f'{idx:06d}.json'.format(idx))
        assert os.path.isfile(label_file)
        return astyx_data_utils.read_label(label_file)
        
        
def rt_matrix(yaw=0,roll=0,pitch =0):
    """
        Returns a 3x3 Rotation Matrix. Angels in degree!
    """
    yaw = yaw #* np.pi / 180
    roll = roll #* np.pi / 180
    pitch = pitch# * np.pi / 180
    c_y = np.cos(yaw)
    s_y = np.sin(yaw)
    c_r = np.cos(roll)
    s_r = np.sin(roll)
    c_p = np.cos(pitch)
    s_p = np.sin(pitch)
    
    # Rotationmatrix
    rot = np.dot(np.dot(np.array([[c_y, - s_y,   0],
                                  [s_y,   c_y,   0],
                                  [0,      0,    1]]),
                        np.array([[c_p,    0,    s_p],
                                  [0,      1,    0],
                                  [-s_p,   0,    c_p]])),
                        np.array([[1,      0,    0],
                                  [0,     c_r, - s_r],
                                  [0,     s_r,   c_r]]))
    return rot

def rotate_points(points, rot_t):
    """
        Input must be of shape N x 3
        Returns the rotated point cloud for a given roation matrix 
        and point cloud.
    """
    points = np.dot(rot_t, points[:,0:3].T).T
    return points
