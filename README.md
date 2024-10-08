# FusionComplexYOLO
This repository contains a PyTorch implementation of [ComplexYOLO](https://arxiv.org/pdf/1803.06199.pdf) using YOLO version 4 and High/Low Level Sensorfusion. It is build to be applied on the LiDAR and RADAR data from the Astyx Dataset. For an implementation for the KITTI dataset see [here](https://github.com/maudzung/Complex-YOLOv4-Pytorch).


<img src="https://github.com/BerensRWU/FusionComplexYOLO/blob/main/images/ComplexYOLO_High.png" width="400" height="800">

<img src="https://github.com/BerensRWU/FusionComplexYOLO/blob/main/images/ComplexYOLO_Low.png" width="400" height="800">

## Requirement

```shell script
pip install -U -r requirements.txt
```

#### Steps
1. Install all requirements
1. Save the Astyx dataset in the folder ```dataset```.(See Section Astyx HiRes).
1. Download the weights for the RADAR and LiDAR detector from the moodle page of the Lecture. 

# Astyx HiRes
The Astyx HiRes is a dataset from Astyx for object detection for autonomous driving. Astyx has a sensor setup consisting of camera, LiDAR, RADAR. Additional information can be found here: [Dataset Paper](https://www.astyx.com/fileadmin/redakteur/dokumente/Automotive_Radar_Dataset_for_Deep_learning_Based_3D_Object_Detection.PDF) and [Specification](https://www.astyx.com/fileadmin/redakteur/dokumente/Astyx_Dataset_HiRes2019_specification.pdf)

```
└── dataset/
       ├── dataset_astyx_hires2019    <-- 546 data
       |   ├── calibration 
       |   ├── camera_front
       |   ├── groundtruth_obj3d
       |   ├── lidar_vlp16
       └── ├── radar_6455 
```
### Training
For Low Level Fusion:
```
python3 train.py \
  --saved_fn 'complexyolov4' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 8 \
  --num_workers 4 \
  --gpu_idx 0 \
  --dataset 'astyx' \
  --checkpoint_freq 300 \
  --low_fusion 
```

For High Level Fusion:
```
python3 train.py \
  --saved_fn 'complexyolov4' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 8 \
  --num_workers 4 \
  --gpu_idx 0 \
  --dataset 'astyx' \
  --checkpoint_freq 300 \
  --lidar

python3 train.py \
  --saved_fn 'complexyolov4' \
  --arch 'darknet' \
  --cfgfile ./config/cfg/complex_yolov4.cfg \
  --batch_size 8 \
  --num_workers 4 \
  --gpu_idx 0 \
  --dataset 'astyx' \
  --checkpoint_freq 300 \
  --radar \
```
### Evaluation
For Low Level Fusion:
```
python3 evaluate.py \
  --pretrained_path path/to/weights/of/low/level/fusion/model \
  --cfgfile './config/cfg/complex_yolov4.cfg'\
  --dataset 'astyx' \
  --classnames-infor-path '../dataset/astyx/classes_names.txt' \
  --gpu_idx 0 \
  --low_fusion \
  --nms-thresh 0.5 \
  --conf-thresh 0.5 \
```
For High Level Fusion:
```
python3 evaluate_high_astyx.py \
  --pretrained_path_lidar path/to/weights/of/low/level/fusion/model \
  --pretrained_path_radar path/to/weights/of/low/level/fusion/model \
  --cfgfile './config/cfg/complex_yolov4.cfg'\
  --dataset 'astyx' \
  --classnames-infor-path '../dataset/astyx/classes_names.txt' \
  --gpu_idx 0 \
  --high_fusion \
  --nms-thresh 0.5 \
  --conf-thresh 0.5
```
