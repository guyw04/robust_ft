import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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




setup_logger()


def get_test_cfg():
    cfg = get_cfg()
    cfg.merge_from_file("/home/alistair/Work/tumour_identification/robust/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.WEIGHTS = "/home/alistair/Work/tumour_identification/robust/detectron2/tools/output_1/mymodel_25000.pth"
    cfg.DATASETS.TEST = ("my_dataset_val", )
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5   # Set a threshold for this model
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.DATALOADER.NUM_WORKERS = 2
    return cfg

register_coco_instances("my_dataset_val", {},
                            "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format.json",
                            "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
# Load the configuration and metadata
cfg = get_test_cfg()


def plot_top_boxes(image, outputs, top_k=3):
    """
    Plot the top K bounding boxes with highest scores using matplotlib.
    """
    scores = outputs['instances'].scores.cpu().numpy()
    boxes = outputs['instances'].pred_boxes.tensor.cpu().numpy()
    classes = outputs['instances'].pred_classes.cpu().numpy()
    class_names = MetadataCatalog.get(cfg.DATASETS.TEST[0]).get("thing_classes", None)

    top_indices = scores.argsort()[-top_k:][::-1]
    top_boxes = boxes[top_indices]

    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(image[:, :, ::-1])

    for i, box in enumerate(top_boxes):
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        if class_names:
            label = class_names[classes[top_indices[i]]]
            score = scores[top_indices[i]]
            ax.text(x1, y1, f'{label}: {score:.2f}', color='white', fontsize=8, bbox=dict(facecolor='red', alpha=0.5))

    # Hide axes
    ax.axis('off')
    plt.show()
    #plt.close()  # Prevents the figure from being displayed here, we only want to return the image.

    # Convert the matplotlib figure to an OpenCV image
    fig.canvas.draw()
    result_image = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    result_image = result_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    return result_image

evaluator = DefaultPredictor(cfg)

# Correctly load the dataset dictionaries
dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])

# Continue with your evaluation and plotting logic
for d in dataset_dicts:
    input(d)
    img_path = d["file_name"]
    img = cv2.imread(img_path)
    outputs = evaluator(img)
    result_img = plot_top_boxes(img, outputs, top_k=3)

    # Save or display the result image
    #result_path = os.path.join("output", os.path.basename(img_path))
    #cv2.imwrite(result_path, result_img)
    #print(f"Processed {img_path}, result saved to {result_path}")

print("Evaluation and plotting complete.")