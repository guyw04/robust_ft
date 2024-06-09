import os
import logging
import torch
import detectron2
import concurrent.futures
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
import numpy as np
import torch
from torch.optim.optimizer import Optimizer, required
import copy
import math
import time

from detectron2.solver.build import maybe_add_gradient_clipping

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
    cfg.SOLVER.IMS_PER_BATCH = 6
    cfg.SOLVER.BASE_LR = 1e-2
    cfg.SOLVER.MAX_ITER = 100000
    cfg.TEST.EVAL_PERIOD = 1000
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
    cfg.OUTPUT_DIR = './output_ftp/'
    #cfg.SOLVER.AMP.ENABLED = False
    return cfg



#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################

class FTP(object):
    def __init__(self, k=1.0, exclude_set={}):
        self.exclude_set = exclude_set
        self.threshold = torch.nn.Hardtanh(0, 1)
        self.k = k  # Gradient annealing factor
        self.j = 0  # Buffer counter

        # AdamUtil parameteres
        self.mu = 1e-2
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.t = 1

        # Buffers
        self.gamma_buff = []
        self.first_m_gamma = []
        self.second_m_gamma = []
        self.prev_c = []
        self.prev_scale = []

    @torch.no_grad()
    def step(self, name, curr, pre, d_p):
        if curr.requires_grad and name not in self.exclude_set:
            c_t = (curr - d_p) - pre
            norms = self._mars_norm(c_t)

            if self.t == 1:
                gamma = torch.tensor(1e-8).to(norms.device)
                self._update_buffers(gamma)
            else:
                # Get previous values
                gamma_prev = self.gamma_buff[self.j]
                c_prev = self.prev_c[self.j]
                scale_prev = self.prev_scale[self.j]

                # Calculate gradient for gamma
                gamma_grad = torch.sum(self._dot(curr.grad, c_prev, scale=scale_prev))

                # Anneal positive gradient
                if gamma_grad > 0:
                    gamma_grad = gamma_grad * self.k

                gamma = self._adam_util(gamma_prev, gamma_grad)

                # Clip gamma
                gamma = self._clip(gamma, norms)

            # Update
            denom = 1 / norms
            ratio = gamma * denom
            new_p = pre + self.threshold(ratio) * c_t

            # Save updated values
            self._update_buffers(gamma, c_t, denom)
            self.j += 1

            return new_p
        else:
            return None

    def incre_counters(self):
        self.t += 1
        self.j = 0

    @torch.no_grad()
    def _mars_norm(self, tensor):
        return torch.sum(torch.abs(tensor), dim=tuple(range(1, tensor.dim())), keepdim=True) + 1e-8

    @torch.no_grad()
    def _clip(self, constraint, norms):
        return torch.nn.functional.hardtanh(constraint, 1e-8, norms.max())

    @torch.no_grad()
    def _dot(self, tensor1, tensor2, scale=1):
        return torch.sum(torch.mul(tensor1, tensor2), dim=tuple(range(1, tensor1.dim())), keepdim=True) * scale

    @torch.no_grad()
    def _adam_util(self, prev, grad):
        first_moment = self.beta1 * self.first_m_gamma[self.j] + (1 - self.beta1) * grad
        second_moment = self.beta2 * self.second_m_gamma[self.j] + (1 - self.beta2) * grad ** 2
        self.first_m_gamma[self.j] = first_moment
        self.second_m_gamma[self.j] = second_moment
        first_moment = first_moment / (1 - self.beta1 ** self.t)
        second_moment = second_moment / (1 - self.beta2 ** self.t)
        return prev - self.mu * first_moment / (torch.sqrt(second_moment) + 1e-8)

    def _update_buffers(self, gamma, c_t=None, denom=None):
        if c_t is None:
            self.first_m_gamma.append(0.0)
            self.second_m_gamma.append(0.0)
            self.gamma_buff.append(gamma)
            self.prev_c.append(0.0)
            self.prev_scale.append(0.0)
        else:
            self.gamma_buff[self.j] = gamma
            self.prev_c[self.j] = c_t
            self.prev_scale[self.j] = denom
class AdamP(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, k=1.0, exclude_set={}):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad)

        super(AdamP, self).__init__(params, defaults)


        # initialize FTP
        self.ftp = FTP(k, exclude_set=exclude_set)


    def __setstate__(self, state):
        super(AdamP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []
            max_exp_avg_sqs = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        if group['amsgrad']:
                            # Maintains max of all exp. moving avg. of sq. grad. values
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            beta1, beta2 = group['betas']
            self.adam(group,
                      exp_avgs,
                      exp_avg_sqs,
                      max_exp_avg_sqs,
                      state_steps,
                      group['amsgrad'],
                      beta1,
                      beta2,
                      group['lr'],
                      group['weight_decay'],
                      group['eps']
                      )
        # FTP increment internal counters
        self.ftp.incre_counters()

        return loss

    def adam(self, group,
             exp_avgs,
             exp_avg_sqs,
             max_exp_avg_sqs,
             state_steps,
             amsgrad: bool,
             beta1: float,
             beta2: float,
             lr: float,
             weight_decay: float,
             eps: float):

        i = 0
        for param, name, pre in zip(group['params'], group['name'], group['pre']):
            if param.grad is None:
                continue
            grad = param.grad
            exp_avg = exp_avgs[i]
            exp_avg_sq = exp_avg_sqs[i]
            step = state_steps[i]
            if amsgrad:
                max_exp_avg_sq = max_exp_avg_sqs[i]

            bias_correction1 = 1 - beta1 ** step
            bias_correction2 = 1 - beta2 ** step

            # Decay the first and second moment running average coefficient
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            if amsgrad:
                # Maintains the maximum of all 2nd moment running avg. till now
                torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                # Use the max. for normalizing running avg. of gradient
                denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
            else:
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

            step_size = lr / bias_correction1
            i += 1

            # FTP step
            d_p = step_size * exp_avg / denom + lr * weight_decay * param
            new_p = self.ftp.step(name, param, pre, d_p)
            if new_p is None:
                new_p = param - d_p
            param.copy_(new_p)


class SGDP(Optimizer):
    def __init__(self, params, lr=required, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False, k=1.0, exclude_set={}):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGDP, self).__init__(params, defaults)
        self.first_iter_flag = False

        # initialize FTP
        self.ftp = FTP(k, exclude_set=exclude_set)

    def __setstate__(self, state):
        super(SGDP, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p, name, pre in zip(group['params'], group['name'], group['pre']):
                if p.grad is None:
                    continue
                d_p = p.grad
                if weight_decay != 0:
                    d_p = d_p.add(p, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)
                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                # FTP step
                d_p = group['lr'] * d_p
                new_p = self.ftp.step(name, p, pre, d_p)
                if new_p is not None:
                    p.copy_(new_p)
                else:
                    p.add_(d_p, alpha=-1)
        # FTP increment internal counters
        self.ftp.incre_counters()
        return loss





#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################
#################################################################################################################




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
                    inputs = [{"image": image, "height": height, "width": width, "instances": annotations[0]}]

                    # Assuming model has a custom implementation to calculate loss during evaluation
                    # This requires the model to accept ground truth in some form and return losses
                    outputs = self.trainer.model(inputs)
                    outputs = outputs[0]
                    if len(outputs["instances"].pred_boxes.tensor.cpu()) > 0:
                        pred_box = outputs["instances"].pred_boxes.tensor.cpu().numpy()[0]  # First box
                        pred_box = BoxMode.convert(np.expand_dims(pred_box, 0), BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
                        gt_box = item["annotations"][0]["bbox"] if item["annotations"] else None
                        # Log the calculated losses
                        iou = calculate_iou(pred_box, gt_box)
                    else:
                        iou = 0.0
                    print(iou)
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
        elif cfg.SOLVER.OPTIMIZER == "AdamP":  # Correct the case to match your intended usage

            optimizer_params = {
                "lr": cfg.SOLVER.BASE_LR,
                "weight_decay": 0.1,
                "k": 0.25,
                "exclude_set": {'module.head.weight', 'module.head.bias'}
            }



            # Cache pre-trained model weights
            params_to_opt = [x[1] for x in model.named_parameters() if x[1].requires_grad]
            params_to_opt_name = [x[0] for x in model.named_parameters() if x[1].requires_grad]
            params_anchor = copy.deepcopy(params_to_opt)
            param_group = [{'params': params_to_opt,
                            'pre': params_anchor,
                            'name': params_to_opt_name}]


            return AdamP(param_group, **optimizer_params)
        elif cfg.SOLVER.OPTIMIZER == "SGDP":
            # Initalize optimizer parameters
            optimizer_params = {
                "lr": cfg.SOLVER.BASE_LR,
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


            return SGDP(param_group, **optimizer_params)
        else:
            raise NotImplementedError(f"{cfg.SOLVER.OPTIMIZER} is not supported")

    """@classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)#cfg.SOLVER.MAX_ITER)"""

    def __init__(self, cfg):
        super().__init__(cfg)

    """def run_step(self):

        assert self.model.training, "[Trainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)  # Use self here
        data_time = time.perf_counter() - start

        if self._trainer.zero_grad_before_forward:  # Use self here
            self._trainer.optimizer.zero_grad()  # Use self here

        loss_dict = self.model(data)
        if isinstance(loss_dict, torch.Tensor):
            losses = loss_dict
            loss_dict = {"total_loss": loss_dict}
        else:
            losses = sum(loss_dict.values())

        if not self._trainer.zero_grad_before_forward:  # Use self here
            self.optimizer.zero_grad()  # Use self here

        losses.backward()

        self._trainer.after_backward()  # This method should be defined in your class or inherited

        if self._trainer.async_write_metrics:  # Use self here
            self._trainer.concurrent_executor.submit(
                self._trainer._write_metrics, loss_dict, data_time, iter=self.iter  # Use self here
            )
        else:
            self._trainer._write_metrics(loss_dict, data_time)  # Use self here

        # Capture weights before the optimization step
        weights_before = {name: param.clone() for name, param in self.model.named_parameters()}

        self.optimizer.step()

        # Capture weights after the optimization step
        weights_after = {name: param for name, param in self.model.named_parameters()}

        # Compute the sum of absolute differences
        total_difference = sum(
            torch.sum(torch.abs(weights_before[name] - weights_after[name])) for name in weights_before)

        #input(f"Sum of absolute differences in weights: {total_difference.item()}")"""




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
            T.RandomFlip(prob=0.25),
        ])  # type: T.Augmentation
        mapper = DatasetMapper(cfg, is_train=True, augmentations=[augs])

        return build_detection_train_loader(cfg, mapper=mapper)


def main():
    cfg = get_cfg_custom()
    cfg.SOLVER.OPTIMIZER = "SGDP"  # Or "Adam"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()

if __name__ == "__main__":
    print('DO NOT USE')
    print('DO NOT USE')
    print('DO NOT USE')
    print('DO NOT USE')
    print('DO NOT USE')
    exit()
    main()
