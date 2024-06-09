import os
import numpy as np
import torch
import datetime
import logging
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, HookBase
from detectron2.data import build_detection_test_loader, DatasetMapper
from detectron2.utils.logger import log_every_n_seconds
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
import detectron2.utils.comm as comm


# Step 1: Dataset Registration
register_coco_instances("my_dataset_train", {}, "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/train_coco_format.json",
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
register_coco_instances("my_dataset_val", {}, "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format.json",
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")

# Step 2: Configuration Setup
cfg = get_cfg()
# For object detection, use a COCO-Detection configuration
cfg.merge_from_file("/home/alistair/Work/tumour_identification/robust/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")  # Use pretrained weights for faster_rcnn
cfg.DATASETS.TRAIN = ("my_dataset_train",)
cfg.DATASETS.TEST = ("my_dataset_val",)
cfg.DATALOADER.NUM_WORKERS = 2

# Solver settings
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.TEST.EVAL_PERIOD = 400
cfg.SOLVER.MAX_ITER = 3000  # Adjust based on dataset size
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Adjust number of classes
cfg.OUTPUT_DIR = './output/'
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

# Step 3: Training
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


