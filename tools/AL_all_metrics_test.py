import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from detectron2.structures import Boxes, BoxMode
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, build_detection_test_loader
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import cv2
from detectron2.data import DatasetCatalog

from collections import defaultdict



setup_logger()


def calculate_iou(boxA, boxB):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.
    Parameters:
        box_a (list): The [x1, y1, x2, y2] coordinates of the first box.
        box_b (list): The [x1, y1, x2, y2] coordinates of the second box.
    Returns:
        float: the IoU metric.
    """
    # Convert boxes from [x_min, y_min, width, height] to [x_min, y_min, x_max, y_max]
    boxA = [boxA[0], boxA[1], boxA[0] + boxA[2], boxA[1] + boxA[3]]
    boxB = [boxB[0], boxB[1], boxB[0] + boxB[2], boxB[1] + boxB[3]]

    # Calculate the intersection coordinates
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Calculate intersection area
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    # Calculate the areas of both bounding boxes
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    # Calculate the union area
    union_area = boxA_area + boxB_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou




def get_test_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/home/alistair/Work/tumour_identification/robust/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.WEIGHTS = "/home/alistair/Work/tumour_identification/robust/detectron2/AL_FT/FTP-main/output_adamw_d2/mymodel_20000.pth"   ### CHANGE HERE !!
    cfg.DATASETS.TEST = ("my_dataset_val_2", )  ### CHANGE HERE !!
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.NUM_WORKERS = 2
    return cfg

register_coco_instances("my_dataset_val_2", {},  ### CHANGE HERE !!
                            "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format_2.json",  ### CHANGE HERE !!
                            "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
# Load the configuration and metadata
cfg = get_test_cfg()


evaluator = DefaultPredictor(cfg)

# Correctly load the dataset dictionaries
dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
iou_scores = []
detection_scores = []
# Continue with your evaluation and plotting logic
for d in dataset_dicts:
    img_path = d["file_name"]
    img = cv2.imread(img_path)
    outputs = evaluator(img)

    # Only use the most confident/first bounding box prediction
    # Assuming there is only one ground truth box and its mode is XYWH_ABS
    gt_box = d["annotations"][0]["bbox"] if d["annotations"] else None

    x_min, y_min, _, _ = gt_box

    if x_min > 1200 and y_min < 60:
        continue

    if len(outputs["instances"].pred_boxes):
        pred_box = outputs["instances"].pred_boxes.tensor.cpu().numpy()[0]  # First box
        confidence = outputs["instances"].scores[0].cpu().numpy()  # Confidence of the first box
        pred_box = BoxMode.convert(np.expand_dims(pred_box,0), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
        pred_box = pred_box[0]


        if gt_box:
            iou_indiv = calculate_iou(pred_box, gt_box)
            print(iou_indiv)
            iou_scores.append(iou_indiv)
            detection_scores.append((iou_indiv, confidence))

    else:
        iou_scores.append(0.0)

print(detection_scores)
iou_scores = np.array(iou_scores)
average_iou = np.mean(iou_scores)

# Now calculate precision and recall for various confidence thresholds
confidence_thresholds = np.linspace(0, 1, 101)  # Adjust the granularity as needed
precisions = []
recalls = []
iou_threshold = 0.5  # Common IoU threshold for considering a detection as a true positive

for conf_threshold in confidence_thresholds:
    tp = 0
    fp = 0
    fn = len(dataset_dicts)  # Assuming one object per image

    for iou, conf in detection_scores:
        if conf >= conf_threshold:
            if iou >= iou_threshold:
                tp += 1
            else:
                fp += 1

    fn -= tp  # Subtract true positives from the total number of objects to get false negatives
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    precisions.append(precision)
    recalls.append(recall)

# Calculate AP as the area under the precision-recall curve
# The following AP calculation method is a simplified version; consider using established libraries for more robust calculation
ap = np.trapz(sorted(precisions), x=sorted(recalls))

print(f"Average IoU: {np.mean([iou for iou, _ in detection_scores])}")
print(f"AP at IoU threshold {iou_threshold}: {ap}") # always going to be low because confidence is always near 1
# because single object single class



######################################################### FASTER RCNN
#COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml
#cfg.DATALOADER.NUM_WORKERS = 2
#cfg.SOLVER.IMS_PER_BATCH = 6
#cfg.SOLVER.BASE_LR = 1e-4
#cfg.SOLVER.MAX_ITER = 20200
#cfg.TEST.EVAL_PERIOD = 5000
#cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
#cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

######################################################### adam
#lr = 1e-4
#"weight_decay" = 5.0e-4
# d1 = 0.480872608941223
# d2 = 0.5854308941498301
# d3 = 0.7095468114790164

######################################################### adam-p
# k = 1
# d1 = 0.6105000242386844
# d2 = 0.6303440282664429
# d3 = 0.6269163995527913

# k = 0.5
# d1 = 0.6190697635050665

######################################################### sgd
#"lr": 1.0e-6,  # cfg.SOLVER.BASE_LR
#"weight_decay": 5.0e-4
#"momentum": 0.9
#"nesterov": True
# d1 = 0.4663795916105169
# d2 = 0.45567977145028254
# d3 = 0.42724281352940624

######################################################### sgd-p
# k = 1
# d1 = 0.5740157342027509
# d2 = 0.5978801179066194
# d3 = 0.5813044335060852

######################################################### adamw
# d1 = 0.5880111483790564
# d2 = 0.562538895255414
# d3 =