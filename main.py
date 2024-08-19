import numpy as np
import sys
import random
import os
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

import torch

from easydict import EasyDict as edict

sys.path.append('./')

from models.model_utils import create_model, make_data_parallel
from utils.train_utils import create_optimizer, create_lr_scheduler
from config.train_config import parse_train_configs
from evaluate import evaluate_mAP
from train import cleanup, train_one_epoch


def main():
    configs = parse_train_configs()

    # Re-produce results
    if configs.seed is not None:
        random.seed(configs.seed)
        np.random.seed(configs.seed)
        torch.manual_seed(configs.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if configs.gpu_idx is not None:
        print('You have chosen a specific GPU. This will completely disable data parallelism.')

    main_worker(configs.gpu_idx, configs)


def main_worker(gpu_idx, configs):
    configs.gpu_idx = gpu_idx
    configs.device = torch.device('cpu' if configs.gpu_idx is None else 'cuda:{}'.format(configs.gpu_idx))
    configs.is_master_node = True
    configs.distributed = False
    
    eval_epoch_start = 150

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


    configs_eval_dist_lidar_fog = edict(configs.copy())
    configs_eval_dist_lidar_fog.disturb_types_training_lidar = ['fog']
    configs_eval_dist_lidar_fog.disturb_types_training_radar = ['None']

    configs_eval_dist_lidar_snow = edict(configs.copy())
    configs_eval_dist_lidar_snow.disturb_types_training_lidar = ['snow']
    configs_eval_dist_lidar_snow.disturb_types_training_radar = ['None']

    if configs.dataset.upper() == "ASTYX":
        from data_process_astyx.astyx_dataloader import create_train_dataloader, create_val_dataloader
    else:
        raise NotImplementedError

    # Dimensions: Repeation x levels x metrics
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
    val_best_dist_lidar_fog = np.zeros(val_list_dim)
    val_best_dist_lidar_snow = np.zeros(val_list_dim)

    for rep in range(configs.repeats):
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

        configs_eval_dist_lidar_fog.set_seed = configs.set_seed
        configs_eval_dist_lidar_fog.subdivisions = int(64 / configs.batch_size)

        configs_eval_dist_lidar_snow.set_seed = configs.set_seed
        configs_eval_dist_lidar_snow.subdivisions = int(64 / configs.batch_size)

        configs.subdivisions = int(64 / configs.batch_size)

        # model
        model = create_model(configs)

        # Data Parallel
        model = make_data_parallel(model, configs)

        # Make sure to create optimizer after moving the model to cuda
        optimizer = create_optimizer(configs, model)
        lr_scheduler = create_lr_scheduler(optimizer, configs)
        configs.step_lr_in_epoch = True if configs.lr_type in ['multi_step'] else False

        # resume optimizer, lr_scheduler from a checkpoint
        if configs.resume_path is not None:
            utils_path = configs.resume_path.replace('Model_', 'Utils_')
            assert os.path.isfile(utils_path), "=> no checkpoint found at '{}'".format(utils_path)
            utils_state_dict = torch.load(utils_path, map_location='cuda:{}'.format(configs.gpu_idx))
            optimizer.load_state_dict(utils_state_dict['optimizer'])
            lr_scheduler.load_state_dict(utils_state_dict['lr_scheduler'])
            configs.start_epoch = utils_state_dict['epoch'] + 1

        # Create dataloader
        train_dataloader, train_sampler = create_train_dataloader(configs)

        best_model = []
        AP_best = 0

        for epoch in range(configs.start_epoch, configs.num_epochs + 1):
            print(f"Repeat: {rep}, Epoch: {epoch}")
            # train for one epoch
            train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, epoch, configs, None, None)
            
            if epoch > eval_epoch_start and epoch % 2 == 0:
                val_dataloader = create_val_dataloader(configs)
                precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_none, None)
                if AP[0] >= AP_best:
                    print(precision, recall, AP, f1, ap_class)
                    model_best_AP = create_model(configs)
                    model_best_AP.load_state_dict(model.state_dict())
                    best_model = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
                    AP_best = AP[0]
            if not configs.step_lr_in_epoch:
                lr_scheduler.step()
        print("Best model performance:", best_model)
        for level_id, level in enumerate([[0.], [0.25], [0.5], [0.75], [1.]]):
            model = make_data_parallel(model_best_AP, configs)

            print(f"Repeat: {rep}, Epoch: {best_model[0]}, Level {level}")

            configs_eval_dist_none.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_none)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_none, None)
            val_best_dist_none[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


            configs_eval_dist_lidar_add_random_noise.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_add_random_noise)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_add_random_noise, None)
            val_best_dist_lidar_add_random_noise[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_radar_add_random_noise.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_radar_add_random_noise)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_add_random_noise, None)
            val_best_dist_radar_add_random_noise[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_both_add_random_noise.disturb_levels_training_lidar = level
            configs_eval_dist_both_add_random_noise.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_both_add_random_noise)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_add_random_noise, None)
            val_best_dist_both_add_random_noise[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


            configs_eval_dist_lidar_lose_random.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_lose_random)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_lose_random, None)
            val_best_dist_lidar_lose_random[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_radar_lose_random.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_radar_lose_random)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_lose_random, None)
            val_best_dist_radar_lose_random[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_both_lose_random.disturb_levels_training_lidar = level
            configs_eval_dist_both_lose_random.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_both_lose_random)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_lose_random, None)
            val_best_dist_both_lose_random[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
            
            #####################################################################################################################
            configs_eval_dist_lidar_random_shift.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_random_shift)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_random_shift, None)
            val_best_dist_lidar_random_shift[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_radar_random_shift.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_radar_random_shift)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_random_shift, None)
            val_best_dist_radar_random_shift[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_both_random_shift.disturb_levels_training_lidar = level
            configs_eval_dist_both_random_shift.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_both_random_shift)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_random_shift, None)
            val_best_dist_both_random_shift[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


            configs_eval_dist_lidar_intensity.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_intensity)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_intensity, None)
            val_best_dist_lidar_intensity[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_radar_intensity.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_radar_intensity)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_intensity, None)
            val_best_dist_radar_intensity[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_both_intensity.disturb_levels_training_lidar = level
            configs_eval_dist_both_intensity.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_both_intensity)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_intensity, None)
            val_best_dist_both_intensity[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
            #######################################


            configs_eval_dist_lidar_blobs.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_blobs)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_blobs, None)
            val_best_dist_lidar_blobs[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_radar_blobs.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_radar_blobs)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_radar_blobs, None)
            val_best_dist_radar_blobs[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_both_blobs.disturb_levels_training_lidar = level
            configs_eval_dist_both_blobs.disturb_levels_training_radar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_both_blobs)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_both_blobs, None)
            val_best_dist_both_blobs[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]


            configs_eval_dist_lidar_fog.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_fog)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_fog, None)
            val_best_dist_lidar_fog[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]

            configs_eval_dist_lidar_snow.disturb_levels_training_lidar = level
            val_dataloader = create_val_dataloader(configs_eval_dist_lidar_snow)
            precision, recall, AP, f1, ap_class = evaluate_mAP(val_dataloader, model, configs_eval_dist_lidar_snow, None)
            val_best_dist_lidar_snow[rep,level_id] = [epoch, precision[0], recall[0], AP[0], f1[0], ap_class[0]]
        
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
        np.save(f"{configs.saved_fn}/eval_best_dist_lidar_fog.npy", val_best_dist_lidar_fog)
        np.save(f"{configs.saved_fn}/eval_best_dist_lidar_snow.npy", val_best_dist_lidar_snow)
        

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        try:
            cleanup()
            sys.exit(0)
        except SystemExit:
            os._exit(0)
