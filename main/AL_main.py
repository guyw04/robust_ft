import os
from datetime import datetime
from util.FTP import SGDP
from util.FTP import AdamP
from util.FTP import AdamWP
from util.FTP_modded import AdamPModded

import detectron2
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.data.datasets import register_coco_instances
from detectron2 import model_zoo
from detectron2.utils.events import EventStorage

import copy
import detectron2.utils.comm as comm
import torch
from util import dump_logs
from torch.utils.tensorboard import SummaryWriter

from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog
from detectron2.data import DatasetMapper, build_detection_train_loader
from detectron2.data import transforms as T

from detectron2.data.transforms import ResizeShortestEdge, RandomFlip, RandomCrop, RandomRotation

import copy

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


from torch.utils.tensorboard import SummaryWriter




def evaluate_model(cfg, model, iteration, writer):
    print('BEGIN EVAL')
    evaluator = COCOEvaluator(cfg.DATASETS.TEST[0], cfg, False, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    inference_result = inference_on_dataset(model, val_loader, evaluator)

    # Process and log evaluation metrics
    for metric_name, metric_value in inference_result.items():
        if isinstance(metric_value, (float, int)):  # Check if the value is numeric
            writer.add_scalar(f"Test/{metric_name}", metric_value, iteration)
        elif isinstance(metric_value, dict):  # If the value is a dictionary, process it further
            for sub_metric_name, sub_metric_value in metric_value.items():
                if isinstance(sub_metric_value, (float, int)):  # Ensure the sub-value is numeric
                    writer.add_scalar(f"Test/{metric_name}_{sub_metric_name}", sub_metric_value, iteration)
    print('END EVAL')


def filter_annotations(annotations):
    filtered_annotations = []
    for annotation in annotations:
        # Extract the bounding box coordinates
        x_min, y_min, bbox_width, bbox_height = annotation['bbox']

        # Criteria for being in the designated "no object" indicator area
        is_top_right_x = x_min > 1200  # Bounding box starts in the far right
        is_top_right_y = y_min < 60  # Bounding box is at the very top

        # Keep the annotation if it doesn't meet both criteria
        if not (is_top_right_x and is_top_right_y):
            filtered_annotations.append(annotation)

    return filtered_annotations

def custom_mapper(dataset_dict):
    # Process a copy of the dataset_dict to avoid modifying the original
    dataset_dict = copy.deepcopy(dataset_dict)

    # Filter out 'no object' indicator annotations
    dataset_dict['annotations'] = filter_annotations(dataset_dict['annotations'])

    augs = T.AugmentationList([
        T.RandomBrightness(0.4, 1.6),
        T.RandomSaturation(0.4, 1.6),
        T.RandomContrast(0.4, 1.6),
        T.RandomFlip(prob=0.5),
    ])  # type: T.Augmentation
    mapper = DatasetMapper(cfg, is_train=True, augmentations=[augs])
    return mapper(dataset_dict)


def custom_train_loader(cfg):
    return build_detection_train_loader(cfg, mapper=custom_mapper)



#register_coco_instances("my_dataset_train", {},
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/train_coco_format.json",
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
#register_coco_instances("my_dataset_val", {},
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format.json",
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")

#register_coco_instances("my_dataset_train_2", {},
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/train_coco_format_2.json",
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
#register_coco_instances("my_dataset_val_2", {},
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format_2.json",
#                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")

register_coco_instances("my_dataset_train_3", {},
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/train_coco_format_3.json",
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
register_coco_instances("my_dataset_val_3", {},
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format_3.json",
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")

cfg = get_cfg()
cfg.merge_from_file(
    "/home/alistair/Work/tumour_identification/robust/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.DATASETS.TRAIN = ("my_dataset_train_3",) ### CHANGE HERE !! '',_2,_3
cfg.DATASETS.TEST = ("my_dataset_val_3",) ### CHANGE HERE !!
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 6
cfg.SOLVER.BASE_LR = 1e-4


cfg.SOLVER.MAX_ITER = 20200
cfg.TEST.EVAL_PERIOD = 5000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = './output_adamw_d3/' ### CHANGE HERE !!
opto = "adamw" ### CHANGE HERE !! sometimes




def train(logdir, cfg):
    torch.manual_seed(0)



    # Initialize a TensorBoard writer
    log_dir = cfg.OUTPUT_DIR + "/tensorboard_logs"  # Specify the directory for TensorBoard logs
    writer = SummaryWriter(log_dir)


    # Build the model
    model = build_model(cfg)
    model.train()


    # Setup optimization parameters
    if opto == "sgdp":
        # Initalize optimizer parameters
        optimizer_params = {
            "lr": cfg.SOLVER.BASE_LR ,
            "weight_decay": 0.0,
            "momentum": 0.9,
            "nesterov": True,
            "k": 1, 
            "exclude_set": {'module.head.weight','module.head.bias'}
        } 
        # Cache pre-trained model weights 
        params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
        params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
        params_anchor = copy.deepcopy(params_to_opt)
        param_group = [{'params':params_to_opt,
                        'pre': params_anchor, 
                        'name': params_to_opt_name}]
        optimizer = SGDP(param_group,**optimizer_params)

    elif opto == "adamp":
        # Initalize optimizer parameters
        optimizer_params = {
            "lr": cfg.SOLVER.BASE_LR ,
            "weight_decay": 5.0e-4,
            "k": 1,
            "exclude_set": {'module.head.weight','module.head.bias'}
        } 

        # Cache pre-trained model weights 
        params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
        params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
        params_anchor = copy.deepcopy(params_to_opt)
        param_group = [{'params':params_to_opt,
                        'pre': params_anchor, 
                        'name': params_to_opt_name}]
        optimizer = AdamP(param_group,**optimizer_params)
    elif opto == "adamp_modded":
        # Initalize optimizer parameters
        optimizer_params = {
            "lr": cfg.SOLVER.BASE_LR ,
            "weight_decay": 5.0e-4,
            "k": 1,
            "exclude_set": {'module.head.weight','module.head.bias'}
        }

        # Cache pre-trained model weights
        params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
        params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
        params_anchor = copy.deepcopy(params_to_opt)
        param_group = [{'params':params_to_opt,
                        'pre': params_anchor,
                        'name': params_to_opt_name}]
        optimizer = AdamPModded(param_group,**optimizer_params)
    elif opto == "adamwp":
        # Initalize optimizer parameters
        optimizer_params = {
            "lr": cfg.SOLVER.BASE_LR,
            "weight_decay": 5.0e-4,
            "k": 1,
            "exclude_set": {'module.head.weight', 'module.head.bias'}
        }

        # Cache pre-trained model weights
        params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
        params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
        params_anchor = copy.deepcopy(params_to_opt)
        param_group = [{'params': params_to_opt,
                        'pre': params_anchor,
                        'name': params_to_opt_name}]
        optimizer = AdamWP(param_group, **optimizer_params)
    elif opto == "sgd":
        optimizer_params = {
            "lr": 1.0e-6,#cfg.SOLVER.BASE_LR ,
            "weight_decay": 5.0e-4,
            "momentum": 0.9,
            "nesterov": True,
        }   
        optimizer = torch.optim.SGD(model.parameters(), **optimizer_params)
    elif opto == "adam":
        optimizer_params = {
            "lr": cfg.SOLVER.BASE_LR ,
            "weight_decay": 5.0e-4,
        }
        optimizer = torch.optim.Adam(model.parameters(), **optimizer_params)
    elif opto == "adamw":
        optimizer_params = {
            "lr": cfg.SOLVER.BASE_LR ,
            "weight_decay": 5.0e-4,
        }
        optimizer = torch.optim.AdamW(model.parameters(), **optimizer_params)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.SOLVER.MAX_ITER)

    data_loader = custom_train_loader(cfg)

    checkpointer = DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR)

    # ================================ Training ==========================================
    start_iter = 0
    model.train()
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, cfg.SOLVER.MAX_ITER)):
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            print(iteration)
            print(losses)
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {
                k: v.item() for k, v in comm.reduce_dict(loss_dict).items()
            }
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

                # Log losses to TensorBoard
                writer.add_scalar("Loss/total_loss", losses_reduced, iteration)
                for loss_name, loss_value in loss_dict_reduced.items():
                    writer.add_scalar(f"Loss/{loss_name}", loss_value, iteration)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            # Log learning rate to TensorBoard
            lr = optimizer.param_groups[0]["lr"]
            storage.put_scalar("lr", lr, smoothing_hint=False)
            writer.add_scalar("Learning_Rate", lr, iteration)
            scheduler.step()

            # Inside your training loop
            if iteration > 0:
                if iteration % cfg.TEST.EVAL_PERIOD == 0 or iteration == cfg.SOLVER.MAX_ITER - 1:
                    # Perform evaluation
                    evaluate_model(cfg, model, iteration, writer)
                    # Continue with checkpoint saving and other training steps...
                    checkpointer.save(f"mymodel_{iteration}")
                    print(f"Checkpoint saved at iteration {iteration}")
                # Checkpoint saving using DetectionCheckpointer
                if iteration % cfg.TEST.EVAL_PERIOD == 0 or iteration == cfg.SOLVER.MAX_ITER - 1:
                    checkpointer.save(f"mymodel_{iteration}")
                    print(f"Checkpoint saved at iteration {iteration}")

    # Don't forget to close the writer when you're done
    writer.close()

if __name__ == "__main__":



    now = datetime.now()
    logdir = cfg.OUTPUT_DIR+"log/{}".format(now.strftime("%d_%m_%Y_%H_%M_%S"))
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    print("RUNDIR: {}".format(logdir))
    train(logdir, cfg)

    # for batch 6 = (2360/6)*52=20453 iterations