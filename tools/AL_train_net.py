import os
import logging
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer, launch, HookBase
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader, build_detection_train_loader, DatasetMapper, DatasetCatalog
from detectron2 import model_zoo
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.logger import setup_logger
from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.solver import build_lr_scheduler, build_optimizer
import detectron2.utils.comm as comm
from detectron2.utils.events import get_event_storage, EventStorage
from detectron2.data import DatasetCatalog, build_detection_test_loader, detection_utils as utils
from detectron2.structures import BoxMode
from detectron2.data import transforms as T
setup_logger()

# Dataset Registration
register_coco_instances("my_dataset_train", {}, "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/train_coco_format.json",
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")
register_coco_instances("my_dataset_val", {}, "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/val_coco_format.json",
                        "/home/alistair/Work/tumour_identification/robust/detectron2/AL_DATA/MICCAI24_DATA_ANNOTATED")

def get_cfg_custom():
    cfg = get_cfg()
    cfg.merge_from_file("/home/alistair/Work/tumour_identification/robust/detectron2/configs/COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    cfg.DATASETS.TRAIN = ("my_dataset_train")
    cfg.DATASETS.TEST = ("my_dataset_val",)
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.SOLVER.IMS_PER_BATCH = 8
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.MAX_ITER = 30000
    cfg.TEST.EVAL_PERIOD = 5000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = './output_bin/'
    return cfg


def calculate_metrics_for_top_prediction(output, gt_instances, iou_threshold=0.5):
    tp, fp, fn = 0, 0, 0

    # Sort the predictions by their confidence score to ensure the top prediction is first
    # This is done by assuming the 'output' is already sorted as per Detectron2's standard practice
    try:
        top_pred_box = output["instances"].pred_boxes.tensor[0].tolist()  # Top prediction bounding box
    except:
        return 0.0
    top_pred_score = output["instances"].scores[0].item()  # Top prediction score

    # Convert gt_instances to a format that can be easily compared with predictions
    gt_boxes = gt_instances.gt_boxes.tensor.tolist()
    gt_boxes = gt_boxes[0]
    boxA = gt_boxes
    boxB = top_pred_box
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

class ValidationLossHook(HookBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg.clone()
        self.dataset_name = cfg.DATASETS.TEST[0]  # Assuming a single dataset for simplicity

        self._data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
        self._dataset_dicts = DatasetCatalog.get(cfg.DATASETS.TEST[0])
        interval = len(self._dataset_dicts) // 50  # Calculate the interval to select 20 frames uniformly
        # Slice the dataset to include frames at uniform intervals
        self._dataset_dicts = self._dataset_dicts[::interval]

        self.last_eval_iter = 0
        self.step_its = 0

    def after_step(self):
        next_iter = self.trainer.iter + 1
        if (next_iter % self.cfg.TEST.EVAL_PERIOD == 0) or (next_iter == self.cfg.SOLVER.MAX_ITER):
            self.trainer.model.eval()
            with torch.no_grad():
                counter = 0  # Initialize counter
                for item in self._dataset_dicts:
                    # Prepare the data
                    image = utils.read_image(item["file_name"], format="BGR")
                    height, width = image.shape[:2]
                    image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                    annotations = [utils.annotations_to_instances(item["annotations"], height, width)]
                    # Convert to Instances
                    gt_instances = utils.annotations_to_instances(item['annotations'], height, width)
                    inputs = [{"image": image, "height": height, "width": width, "instances": annotations[0]}]

                    # Assuming model has a custom implementation to calculate loss during evaluation
                    # This requires the model to accept ground truth in some form and return losses
                    outputs = self.trainer.model(inputs)
                    # Log the calculated losses
                    iou = calculate_metrics_for_top_prediction(outputs[0], gt_instances)
                    #print(iou)
                    counter +=1

                    self.trainer.storage.put_scalar("VALIDATION/iou", iou)
                    #with EventStorage(next_iter + counter):
                    #    self.trainer.storage.put_scalar("VALIDATION/iou", iou)

            self.trainer.model.train()  # Set model back to training mode
            checkpointer = DetectionCheckpointer(self.trainer.model, save_dir=self.cfg.OUTPUT_DIR)
            checkpointer.save("mymodel_"+str(next_iter))

class InitializeValidationMetricsHook(HookBase):
    def __init__(self, ):
        pass

    def before_train(self):
        self.storage = get_event_storage()

        self.storage.put_scalar("VALIDATION/iou", 0.0)

class Trainer(DefaultTrainer):

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.SOLVER.OPTIMIZER == "SGD":
            return torch.optim.SGD(model.parameters(), lr=cfg.SOLVER.BASE_LR, momentum=0.9)
        elif cfg.SOLVER.OPTIMIZER == "Adam":
            return torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.BASE_LR)
        else:
            raise NotImplementedError(f"{cfg.SOLVER.OPTIMIZER} is not supported")

    def __init__(self, cfg):
        super().__init__(cfg)


    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(0, InitializeValidationMetricsHook())
        hooks.append(ValidationLossHook(self.cfg))
        return hooks

    def build_train_loader(cls, cfg):

        augs = T.AugmentationList([
            T.RandomBrightness(0.4, 1.6),
            T.RandomSaturation(0.4, 1.6),
            T.RandomContrast(0.4, 1.6),
            T.RandomFlip(prob=0.5),
        ])  # type: T.Augmentation
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[augs])

        return build_detection_train_loader(cfg, mapper=mapper)



def main():
    cfg = get_cfg_custom()
    cfg.SOLVER.OPTIMIZER = "Adam"  # Or "Adam"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    main()
