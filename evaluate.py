import argparse
import os
import time
import numpy as np
import sys
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch
import torch.utils.data.distributed
from easydict import EasyDict as edict
from data_process_astyx.astyx_dataloader_multi import create_val_dataloader

sys.path.append('./')

from models.model_utils import create_model
from utils.misc import AverageMeter, ProgressMeter
from utils.evaluation_utils import  get_batch_statistics_rotated_bbox, ap_per_class, post_processing_v2

def evaluate_mAP(val_loader, model, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(val_loader):
            data_time.update(time.time() - start_time)
            _, imgs, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs = imgs.to(configs.device, non_blocking=True)

            outputs = model(imgs)
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)
            stats = get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)

            sample_metrics += stats if stats else [[np.array([]), torch.tensor([]), torch.tensor([])]]
            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()
        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def evaluate_mAP_high_level(val_loader, model_radar, model_lidar, configs, logger):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')

    progress = ProgressMeter(len(val_loader), [batch_time, data_time],
                             prefix="Evaluation phase...")
    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    sample_metrics1 = []
    sample_metrics2 = []
    # switch to evaluate mode
    model_radar.eval()
    model_lidar.eval()
    with torch.no_grad():
        start_time = time.time()
        for batch_idx, batch_data in enumerate(val_loader):
            data_time.update(time.time() - start_time)
            _, imgs_radar, imgs_lidar, targets = batch_data
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale x, y, w, h of targets ((box_idx, class, x, y, w, l, im, re))
            targets[:, 2:6] *= configs.img_size
            imgs_radar = imgs_radar.to(configs.device, non_blocking=True)
            imgs_lidar = imgs_lidar.to(configs.device, non_blocking=True)

            outputs_radar = model_radar(imgs_radar)
            outputs_lidar = model_lidar(imgs_lidar)
            outputs = []
            for b_id in range(len(outputs_lidar)):
                outputs += [torch.cat((outputs_radar[b_id], outputs_lidar[b_id]))]
            
            outputs = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh)

            stats = get_batch_statistics_rotated_bbox(outputs, targets, iou_threshold=configs.iou_thresh)

            sample_metrics += stats if stats else [[np.array([]), torch.tensor([]), torch.tensor([])]]
            # measure elapsed time
            # torch.cuda.synchronize()
            batch_time.update(time.time() - start_time)

            # Log message
            if logger is not None:
                if ((batch_idx + 1) % configs.print_freq) == 0:
                    logger.info(progress.get_message(batch_idx))

            start_time = time.time()

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)
        
        
    return precision, recall, AP, f1, ap_class


def parse_eval_configs():
    parser = argparse.ArgumentParser(description='Demonstration config for Complex YOLO Implementation')
    
    parser.add_argument('--dataset', type=str, default='KITTI',
                        help='Choose the Dataset. Available:'
                             '   - KITTI'
                             '   - astyx'
                             '   - radiate')
                             
    parser.add_argument('--radar', action='store_true',
                        help='Use RADAR data instead of LiDAR. Only possible'
                            'for Astyx and Radiate')
                            
    parser.add_argument('--saved_fn', type=str, default=None)
                            
    parser.add_argument('--set_seed', type=int, default=1)
    
    parser.add_argument('--repeats', type=int, default=1)
                            
    parser.add_argument('--lidar', action='store_true',
                        help='Use LiDAR-only data.')
                            
    parser.add_argument('--high_fusion', action='store_true',
                        help='High Level Fusion using RADAR and LiDAR data.'
                             'Only possible for Astyx and Radiate.')
                            
    parser.add_argument('--low_fusion', action='store_true',
                        help='Low Level Fusion using RADAR and LiDAR data.'
                             'Only possible for Astyx and Radiate.')

    parser.add_argument('--VR', action='store_true',
                        help='Use the radial velocity from the RADAR data.')

    parser.add_argument('--disturb_types_training_lidar', type=str, default="None")
    parser.add_argument('--disturb_levels_training_lidar', type=str, default="0")

    parser.add_argument('--disturb_types_training_radar', type=str, default="None")
    parser.add_argument('--disturb_levels_training_radar', type=str, default="0")

    parser.add_argument('--classnames-infor-path', type=str, default='../dataset/kitti/classes_names.txt',
                        metavar='PATH', help='The class names of objects in the task')
    parser.add_argument('-a', '--arch', type=str, default='darknet', metavar='ARCH',
                        help='The name of the model architecture')
    parser.add_argument('--cfgfile', type=str, default='./config/cfg/complex_yolov4.cfg', metavar='PATH',
                        help='The path for cfgfile (only for darknet)')

    parser.add_argument('--pretrained_path_radar', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--pretrained_path_lidar', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
    parser.add_argument('--pretrained_path', type=str, default=None, metavar='PATH',
                        help='the path of the pretrained checkpoint')
                        
    parser.add_argument('--use_giou_loss', action='store_true',
                        help='If true, use GIoU loss during training. If false, use MSE loss for training')

    parser.add_argument('--no_cuda', action='store_true',
                        help='If true, cuda is not used.')
    parser.add_argument('--gpu_idx', default=None, type=int,
                        help='GPU index to use.')

    parser.add_argument('--img_size', type=int, default=608,
                        help='the size of input image')
    parser.add_argument('--num_samples', type=int, default=107,
                        help='Take a subset of the dataset to run and debug')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of threads for loading data')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='mini-batch size (default: 4)')

    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for class conf')
    parser.add_argument('--nms-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for nms')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                        help='for evaluation - the threshold for IoU')

    parser.add_argument('--plot_AP', action='store_true',
                        help='Plot the Average Precision of the first class.')
    
    
    
    configs = edict(vars(parser.parse_args()))
    configs.pin_memory = True

    configs.disturb_types_training_lidar = configs.disturb_types_training_lidar.split(" ")
    configs.disturb_levels_training_lidar = configs.disturb_levels_training_lidar.split(" ")
    configs.disturb_levels_training_lidar = [float(x) for x in configs.disturb_levels_training_lidar]


    configs.disturb_types_training_radar = configs.disturb_types_training_radar.split(" ")
    configs.disturb_levels_training_radar = configs.disturb_levels_training_radar.split(" ")
    configs.disturb_levels_training_radar = [float(x) for x in configs.disturb_levels_training_radar]

    ####################################################################
    ##############Dataset, Checkpoints, and results dir configs#########
    ####################################################################
    configs.working_dir = '../'
    if configs.dataset.upper() == 'KITTI':
        configs.dataset_dir = os.path.join(configs.working_dir, 'KITTI')
    if configs.dataset.upper() == 'ASTYX':
        configs.dataset_dir = os.path.join(configs.working_dir, 'dataset_astyx_hires2019')
    if configs.dataset.upper() == 'RADIATE':
        configs.dataset_dir = os.path.join(configs.working_dir, 'radiate', 'data')
    return configs


if __name__ == '__main__':
        
    configs = parse_eval_configs()
    configs.distributed = False  # For evaluation

    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
        
    if configs.high_fusion:
        sensor = 'high_fusion'
    elif configs.radar:
        sensor = "radar"
    else:
        sensor = 'lidar'
        if configs.VR:
            raise ValueError
    
    if sensor != 'lidar':
        if configs.VR:
            sensor += "_VR"
        else:
            sensor += "_Mag"
            
    os.makedirs(configs.saved_fn,exist_ok = True)
    configs_eval_dist_none = edict(configs.copy())
    configs_eval_dist_none.disturb_types_training_lidar = ['None']
    configs_eval_dist_none.disturb_types_training_radar = ['None']


    configs_eval_dist_lidar_add_random_noise = edict(configs.copy())
    configs_eval_dist_lidar_add_random_noise.disturb_types_training_lidar = ['random_noise']
    configs_eval_dist_lidar_add_random_noise.disturb_types_training_radar = ['None']
    
    configs_eval_dist_radar_add_random_noise = edict(configs.copy())
    configs_eval_dist_radar_add_random_noise.disturb_types_training_lidar = ['None']
    configs_eval_dist_radar_add_random_noise.disturb_types_training_radar = ['random_noise']

    configs_eval_dist_both_add_random_noise = edict(configs.copy())
    configs_eval_dist_both_add_random_noise.disturb_types_training_lidar = ['random_noise']
    configs_eval_dist_both_add_random_noise.disturb_types_training_radar = ['random_noise']


    configs_eval_dist_lidar_lose_random = edict(configs.copy())
    configs_eval_dist_lidar_lose_random.disturb_types_training_lidar = ['random_loss']
    configs_eval_dist_lidar_lose_random.disturb_types_training_radar = ['None']

    configs_eval_dist_radar_lose_random = edict(configs.copy())
    configs_eval_dist_radar_lose_random.disturb_types_training_lidar = ['None']
    configs_eval_dist_radar_lose_random.disturb_types_training_radar = ['random_loss']

    configs_eval_dist_both_lose_random = edict(configs.copy())
    configs_eval_dist_both_lose_random.disturb_types_training_lidar = ['random_loss']
    configs_eval_dist_both_lose_random.disturb_types_training_radar = ['random_loss']
    
    ####################################
    configs_eval_dist_lidar_random_shift = edict(configs.copy())
    configs_eval_dist_lidar_random_shift.disturb_types_training_lidar = ['random_shift']
    configs_eval_dist_lidar_random_shift.disturb_types_training_radar = ['None']
    
    configs_eval_dist_radar_random_shift = edict(configs.copy())
    configs_eval_dist_radar_random_shift.disturb_types_training_lidar = ['None']
    configs_eval_dist_radar_random_shift.disturb_types_training_radar = ['random_shift']

    configs_eval_dist_both_random_shift = edict(configs.copy())
    configs_eval_dist_both_random_shift.disturb_types_training_lidar = ['random_shift']
    configs_eval_dist_both_random_shift.disturb_types_training_radar = ['random_shift']


    configs_eval_dist_lidar_intensity = edict(configs.copy())
    configs_eval_dist_lidar_intensity.disturb_types_training_lidar = ['intensity']
    configs_eval_dist_lidar_intensity.disturb_types_training_radar = ['None']

    configs_eval_dist_radar_intensity = edict(configs.copy())
    configs_eval_dist_radar_intensity.disturb_types_training_lidar = ['None']
    configs_eval_dist_radar_intensity.disturb_types_training_radar = ['intensity']

    configs_eval_dist_both_intensity = edict(configs.copy())
    configs_eval_dist_both_intensity.disturb_types_training_lidar = ['intensity']
    configs_eval_dist_both_intensity.disturb_types_training_radar = ['intensity']
    ###############################
    
    
    configs_eval_dist_lidar_blobs = edict(configs.copy())
    configs_eval_dist_lidar_blobs.disturb_types_training_lidar = ['blobs']
    configs_eval_dist_lidar_blobs.disturb_types_training_radar = ['None']

    configs_eval_dist_radar_blobs = edict(configs.copy())
    configs_eval_dist_radar_blobs.disturb_types_training_lidar = ['None']
    configs_eval_dist_radar_blobs.disturb_types_training_radar = ['blobs']

    configs_eval_dist_both_blobs = edict(configs.copy())
    configs_eval_dist_both_blobs.disturb_types_training_lidar = ['blobs']
    configs_eval_dist_both_blobs.disturb_types_training_radar = ['blobs']
    
    configs.set_seed = np.random.randint(10000)

    configs_eval_dist_none.set_seed = configs.set_seed
    configs_eval_dist_none.subdivisions = int(64 / configs.batch_size)

    configs_eval_dist_lidar_add_random_noise.set_seed = configs.set_seed
    configs_eval_dist_lidar_add_random_noise.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_radar_add_random_noise.set_seed = configs.set_seed
    configs_eval_dist_radar_add_random_noise.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_both_add_random_noise.set_seed = configs.set_seed
    configs_eval_dist_both_add_random_noise.subdivisions = int(64 / configs.batch_size)
    
    ################
    configs_eval_dist_lidar_random_shift.set_seed = configs.set_seed
    configs_eval_dist_lidar_random_shift.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_radar_random_shift.set_seed = configs.set_seed
    configs_eval_dist_radar_random_shift.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_both_random_shift.set_seed = configs.set_seed
    configs_eval_dist_both_random_shift.subdivisions = int(64 / configs.batch_size)
    
    configs_eval_dist_lidar_intensity.set_seed = configs.set_seed
    configs_eval_dist_lidar_intensity.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_radar_intensity.set_seed = configs.set_seed
    configs_eval_dist_radar_intensity.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_both_intensity.set_seed = configs.set_seed
    configs_eval_dist_both_intensity.subdivisions = int(64 / configs.batch_size)
    ##################

    configs_eval_dist_lidar_lose_random.set_seed = configs.set_seed
    configs_eval_dist_lidar_lose_random.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_radar_lose_random.set_seed = configs.set_seed
    configs_eval_dist_radar_lose_random.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_both_lose_random.set_seed = configs.set_seed
    configs_eval_dist_both_lose_random.subdivisions = int(64 / configs.batch_size)

    configs_eval_dist_lidar_blobs.set_seed = configs.set_seed
    configs_eval_dist_lidar_blobs.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_radar_blobs.set_seed = configs.set_seed
    configs_eval_dist_radar_blobs.subdivisions = int(64 / configs.batch_size)
    configs_eval_dist_both_blobs.set_seed = configs.set_seed
    configs_eval_dist_both_blobs.subdivisions = int(64 / configs.batch_size)

    configs.subdivisions = int(64 / configs.batch_size)
    
    
    if sensor.split("_")[0] == 'high':
        model_radar = create_model(configs)
        model_lidar = create_model(configs)
        print('\n\n' + '-*=' * 30 + '\n\n')
        #assert os.path.isdir(configs.pretrained_path), f'No dir at {configs.pretrained_path}'
        val_list_dim = (configs.repeats, 5, 6)
        val_best_dist_none = np.zeros(val_list_dim)
        val_best_dist_lidar_add_random_noise = np.zeros(val_list_dim)
        val_best_dist_radar_add_random_noise = np.zeros(val_list_dim)
        val_best_dist_both_add_random_noise = np.zeros(val_list_dim)
        val_best_dist_lidar_lose_random = np.zeros(val_list_dim)
        val_best_dist_radar_lose_random = np.zeros(val_list_dim)
        val_best_dist_both_lose_random = np.zeros(val_list_dim)
        ##################
        val_best_dist_lidar_random_shift = np.zeros(val_list_dim)
        val_best_dist_radar_random_shift = np.zeros(val_list_dim)
        val_best_dist_both_random_shift = np.zeros(val_list_dim)
        val_best_dist_lidar_intensity = np.zeros(val_list_dim)
        val_best_dist_radar_intensity = np.zeros(val_list_dim)
        val_best_dist_both_intensity = np.zeros(val_list_dim)
        ########################
        val_best_dist_lidar_blobs = np.zeros(val_list_dim)
        val_best_dist_radar_blobs = np.zeros(val_list_dim)
        val_best_dist_both_blobs = np.zeros(val_list_dim)
        
        try:
            model_radar.load_state_dict(torch.load(configs.pretrained_path_radar, map_location=torch.device(configs.device)))
            model_radar = model_radar.to(device=configs.device)
            model_radar.eval()
            
            model_lidar.load_state_dict(torch.load(configs.pretrained_path_lidar, map_location=torch.device(configs.device)))
            model_lidar = model_lidar.to(device=configs.device)
            model_lidar.eval()
            
            for level_id, level in enumerate([[0.], [0.25], [0.5], [0.75], [1.]]):
                configs_eval_dist_none.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_none)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_none, None)
                val_best_dist_none[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_lidar_add_random_noise.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_add_random_noise)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_lidar_add_random_noise, None)
                val_best_dist_lidar_add_random_noise[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_add_random_noise.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_add_random_noise)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_radar_add_random_noise, None)
                val_best_dist_radar_add_random_noise[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_add_random_noise.disturb_levels_training_lidar = level
                configs_eval_dist_both_add_random_noise.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_add_random_noise)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_both_add_random_noise, None)
                val_best_dist_both_add_random_noise[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


                configs_eval_dist_lidar_lose_random.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_lose_random)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_lidar_lose_random, None)
                val_best_dist_lidar_lose_random[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_lose_random.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_lose_random)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_radar_lose_random, None)
                val_best_dist_radar_lose_random[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_lose_random.disturb_levels_training_lidar = level
                configs_eval_dist_both_lose_random.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_lose_random)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_both_lose_random, None)
                val_best_dist_both_lose_random[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
                
                #####################################################################################################################
                configs_eval_dist_lidar_random_shift.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_random_shift)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_lidar_random_shift, None)
                val_best_dist_lidar_random_shift[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_random_shift.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_random_shift)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_radar_random_shift, None)
                val_best_dist_radar_random_shift[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_random_shift.disturb_levels_training_lidar = level
                configs_eval_dist_both_random_shift.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_random_shift)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_both_random_shift, None)
                val_best_dist_both_random_shift[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


                configs_eval_dist_lidar_intensity.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_intensity)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_lidar_intensity, None)
                val_best_dist_lidar_intensity[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_intensity.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_intensity)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_radar_intensity, None)
                val_best_dist_radar_intensity[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_intensity.disturb_levels_training_lidar = level
                configs_eval_dist_both_intensity.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_intensity)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_both_intensity, None)
                val_best_dist_both_intensity[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
                #######################################


                configs_eval_dist_lidar_blobs.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_blobs)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_lidar_blobs, None)
                val_best_dist_lidar_blobs[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_blobs.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_blobs)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_radar_blobs, None)
                val_best_dist_radar_blobs[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_blobs.disturb_levels_training_lidar = level
                configs_eval_dist_both_blobs.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_blobs)
                precision, recall, AP, f1, ap_class = evaluate_mAP_high_level(val_dataloader, model_radar, model_lidar, configs_eval_dist_both_blobs, None)
                val_best_dist_both_blobs[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
            np.save(f"{configs.saved_fn}/eval_dist_none.npy", val_best_dist_none)
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_add_random_noise.npy", val_best_dist_lidar_add_random_noise)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_add_random_noise.npy", val_best_dist_radar_add_random_noise)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_add_random_noise.npy", val_best_dist_both_add_random_noise)
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_lose_random.npy", val_best_dist_lidar_lose_random)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_lose_random.npy", val_best_dist_radar_lose_random)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_lose_random.npy", val_best_dist_both_lose_random)
            #################
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_random_shift.npy", val_best_dist_lidar_random_shift)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_random_shift.npy", val_best_dist_radar_random_shift)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_random_shift.npy", val_best_dist_both_random_shift)
            
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_intensity.npy", val_best_dist_lidar_intensity)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_intensity.npy", val_best_dist_radar_intensity)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_intensity.npy", val_best_dist_both_intensity)
            ####################
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_blobs.npy", val_best_dist_lidar_blobs)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_blobs.npy", val_best_dist_radar_blobs)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_blobs.npy", val_best_dist_both_blobs)
        finally:
            pass
        
    else:
        model = create_model(configs)

        print('\n\n' + '-*=' * 30 + '\n\n')
        #assert os.path.isdir(configs.pretrained_path), f'No dir at {configs.pretrained_path}'

        val_list_dim = (configs.repeats, 5, 6)
        val_best_dist_none = np.zeros(val_list_dim)
        val_best_dist_lidar_add_random_noise = np.zeros(val_list_dim)
        val_best_dist_radar_add_random_noise = np.zeros(val_list_dim)
        val_best_dist_both_add_random_noise = np.zeros(val_list_dim)
        val_best_dist_lidar_lose_random = np.zeros(val_list_dim)
        val_best_dist_radar_lose_random = np.zeros(val_list_dim)
        val_best_dist_both_lose_random = np.zeros(val_list_dim)
        ##################
        val_best_dist_lidar_random_shift = np.zeros(val_list_dim)
        val_best_dist_radar_random_shift = np.zeros(val_list_dim)
        val_best_dist_both_random_shift = np.zeros(val_list_dim)
        val_best_dist_lidar_intensity = np.zeros(val_list_dim)
        val_best_dist_radar_intensity = np.zeros(val_list_dim)
        val_best_dist_both_intensity = np.zeros(val_list_dim)
        ########################
        val_best_dist_lidar_blobs = np.zeros(val_list_dim)
        val_best_dist_radar_blobs = np.zeros(val_list_dim)
        val_best_dist_both_blobs = np.zeros(val_list_dim)
        try:
            model.load_state_dict(torch.load(configs.pretrained_path, map_location=torch.device(configs.device)))
            model = model.to(device=configs.device)
            model.eval()

            for level_id, level in enumerate([[0.], [0.25], [0.5], [0.75], [1.]]):

                configs_eval_dist_none.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_none)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_none, None)
                val_best_dist_none[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


                configs_eval_dist_lidar_add_random_noise.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_add_random_noise)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_add_random_noise, None)
                val_best_dist_lidar_add_random_noise[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_add_random_noise.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_add_random_noise)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_add_random_noise, None)
                val_best_dist_radar_add_random_noise[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_add_random_noise.disturb_levels_training_lidar = level
                configs_eval_dist_both_add_random_noise.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_add_random_noise)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_add_random_noise, None)
                val_best_dist_both_add_random_noise[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


                configs_eval_dist_lidar_lose_random.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_lose_random)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_lose_random, None)
                val_best_dist_lidar_lose_random[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_lose_random.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_lose_random)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_lose_random, None)
                val_best_dist_radar_lose_random[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_lose_random.disturb_levels_training_lidar = level
                configs_eval_dist_both_lose_random.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_lose_random)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_lose_random, None)
                val_best_dist_both_lose_random[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                #####################################################################################################################
                configs_eval_dist_lidar_random_shift.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_random_shift)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_random_shift, None)
                val_best_dist_lidar_random_shift[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_random_shift.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_random_shift)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_random_shift, None)
                val_best_dist_radar_random_shift[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_random_shift.disturb_levels_training_lidar = level
                configs_eval_dist_both_random_shift.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_random_shift)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_random_shift, None)
                val_best_dist_both_random_shift[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


                configs_eval_dist_lidar_intensity.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_intensity)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_intensity, None)
                val_best_dist_lidar_intensity[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_intensity.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_intensity)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_intensity, None)
                val_best_dist_radar_intensity[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_intensity.disturb_levels_training_lidar = level
                configs_eval_dist_both_intensity.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_intensity)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_intensity, None)
                val_best_dist_both_intensity[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
                #######################################


                configs_eval_dist_lidar_blobs.disturb_levels_training_lidar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_lidar_blobs)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_blobs, None)
                val_best_dist_lidar_blobs[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_radar_blobs.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_radar_blobs)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_blobs, None)
                val_best_dist_radar_blobs[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

                configs_eval_dist_both_blobs.disturb_levels_training_lidar = level
                configs_eval_dist_both_blobs.disturb_levels_training_radar = level
                val_dataloader = create_val_dataloader(configs_eval_dist_both_blobs)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_blobs, None)
                val_best_dist_both_blobs[0,level_id] = [-1, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            np.save(f"{configs.saved_fn}/eval_dist_none.npy", val_best_dist_none)
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_add_random_noise.npy", val_best_dist_lidar_add_random_noise)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_add_random_noise.npy", val_best_dist_radar_add_random_noise)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_add_random_noise.npy", val_best_dist_both_add_random_noise)
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_lose_random.npy", val_best_dist_lidar_lose_random)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_lose_random.npy", val_best_dist_radar_lose_random)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_lose_random.npy", val_best_dist_both_lose_random)
            #################
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_random_shift.npy", val_best_dist_lidar_random_shift)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_random_shift.npy", val_best_dist_radar_random_shift)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_random_shift.npy", val_best_dist_both_random_shift)

            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_intensity.npy", val_best_dist_lidar_intensity)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_intensity.npy", val_best_dist_radar_intensity)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_intensity.npy", val_best_dist_both_intensity)
            ####################
            np.save(f"{configs.saved_fn}/eval_best_dist_lidar_blobs.npy", val_best_dist_lidar_blobs)
            np.save(f"{configs.saved_fn}/eval_best_dist_radar_blobs.npy", val_best_dist_radar_blobs)
            np.save(f"{configs.saved_fn}/eval_best_dist_both_blobs.npy", val_best_dist_both_blobs)
        finally:
            pass
