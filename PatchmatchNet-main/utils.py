from os import device_encoding
from typing import Any, Callable, Union, Dict

import numpy as np

import torchvision.utils as vutils
import torch
import torch.utils.tensorboard as tb

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# .cuda()  -->(修改为) .to(device)

def print_args(args: Any) -> None:
    """Utilities to print arguments

    Arsg:
        args: arguments to pring out
    """
    print("################################  args  ################################")
    for k, v in args.__dict__.items():
        print("{0: <10}\t{1: <30}\t{2: <20}".format(k, str(v), str(type(v))))
    print("########################################################################")


def make_nograd_func(func: Callable) -> Callable:
    """Utilities to make function no gradient

    Args:
        func: input function

    Returns:
        no gradient function wrapper for input function
    """

    def wrapper(*f_args, **f_kwargs):
        with torch.no_grad():
            ret = func(*f_args, **f_kwargs)
        return ret

    return wrapper


def make_recursive_func(func: Callable) -> Callable:
    """Convert a function into recursive style to handle nested dict/list/tuple variables

    Args:
        func: input function

    Returns:
        recursive style function
    """

    def wrapper(vars):
        if isinstance(vars, list):
            return [wrapper(x) for x in vars]
        elif isinstance(vars, tuple):
            return tuple([wrapper(x) for x in vars])
        elif isinstance(vars, dict):
            return {k: wrapper(v) for k, v in vars.items()}
        else:
            return func(vars)

    return wrapper


@make_recursive_func
def tensor2float(vars: Any) -> float:
    """Convert tensor to float"""
    if isinstance(vars, float):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.data.item()
    else:
        raise NotImplementedError("invalid input type {} for tensor2float".format(type(vars)))


@make_recursive_func
def tensor2numpy(vars: Any) -> np.ndarray:
    """Convert tensor to numpy array"""
    if isinstance(vars, np.ndarray):
        return vars
    elif isinstance(vars, torch.Tensor):
        return vars.detach().cpu().numpy().copy()
    else:
        raise NotImplementedError("invalid input type {} for tensor2numpy".format(type(vars)))


@make_recursive_func
def tocuda(vars: Any) -> Union[str, torch.Tensor]:
    """Convert tensor to tensor on GPU"""
    if isinstance(vars, torch.Tensor):
        return vars.to(device)
    elif isinstance(vars, str):
        return vars
    else:
        raise NotImplementedError("invalid input type {} for tocuda".format(type(vars)))


def save_scalars(logger: tb.SummaryWriter, mode: str, scalar_dict: Dict[str, Any], global_step: int) -> None:
    """Log values stored in the scalar dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        scalar_dict: python dictionary stores the key and value pairs to be recorded
        global_step: step index where the logger should write
    """
    scalar_dict = tensor2float(scalar_dict)
    for key, value in scalar_dict.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_scalar(name, value, global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_scalar(name, value[idx], global_step)


def save_images(logger: tb.SummaryWriter, mode: str, images_dict: Dict[str, Any], global_step: int) -> None:
    """Log images stored in the image dictionary

    Args:
        logger: tensorboard summary writer
        mode: mode name used in writing summaries
        images_dict: python dictionary stores the key and image pairs to be recorded
        global_step: step index where the logger should write
    """
    images_dict = tensor2numpy(images_dict)

    def preprocess(name, img):
        if not (len(img.shape) == 3 or len(img.shape) == 4):
            raise NotImplementedError("invalid img shape {}:{} in save_images".format(name, img.shape))
        if len(img.shape) == 3:
            img = img[:, np.newaxis, :, :]
        img = torch.from_numpy(img[:1])
        return vutils.make_grid(img, padding=0, nrow=1, normalize=True, scale_each=True)

    for key, value in images_dict.items():
        if not isinstance(value, (list, tuple)):
            name = "{}/{}".format(mode, key)
            logger.add_image(name, preprocess(name, value), global_step)
        else:
            for idx in range(len(value)):
                name = "{}/{}_{}".format(mode, key, idx)
                logger.add_image(name, preprocess(name, value[idx]), global_step)


class DictAverageMeter:
    """Wrapper class for dictionary variables that require the average value"""

    def __init__(self) -> None:
        """Initialization method"""
        self.data: Dict[Any, float] = {}
        self.count = 0

    def update(self, new_input: Dict[Any, float]) -> None:
        """Update the stored dictionary with new input data

        Args:
            new_input: new data to update self.data
        """
        self.count += 1
        if len(self.data) == 0:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] = v
        else:
            for k, v in new_input.items():
                if not isinstance(v, float):
                    raise NotImplementedError("invalid data {}: {}".format(k, type(v)))
                self.data[k] += v

    def mean(self) -> Any:
        """Return the average value of values stored in self.data"""
        return {k: v / self.count for k, v in self.data.items()}


def compute_metrics_for_each_image(metric_func: Callable) -> Callable:
    """A wrapper to compute metrics for each image individually"""

    def wrapper(depth_est, depth_gt, mask, *args):
        batch_size = depth_gt.shape[0]
        print(batch_size)
        # if batch_size < BATCH_SIZE:
        #     break
        results = []
        # compute result one by one
        for idx in range(batch_size):
            ret = metric_func(depth_est[idx], depth_gt[idx], mask[idx], *args)
            results.append(ret)
        return torch.stack(results).mean()

    return wrapper


@make_nograd_func
@compute_metrics_for_each_image
def Thres_metrics(
    depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor, thres: Union[int, float]
) -> torch.Tensor:
    """Return error rate for where absolute error is larger than threshold.

    Args:
        depth_est: estimated depth map
        depth_gt: ground truth depth map
        mask: mask
        thres: threshold

    Returns:
        error rate: error rate of the depth map
    """
    # if thres is int or float, then True
    assert isinstance(thres, (int, float))
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    errors = torch.abs(depth_est - depth_gt)
    err_mask = errors > thres
    return torch.mean(err_mask.float())


# NOTE: please do not use this to build up training loss
@make_nograd_func
@compute_metrics_for_each_image
def AbsDepthError_metrics(depth_est: torch.Tensor, depth_gt: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Calculate average absolute depth error

    Args:
        depth_est: estimated depth map
        depth_gt: ground truth depth map
        mask: mask
    """
    depth_est, depth_gt = depth_est[mask], depth_gt[mask]
    return torch.mean((depth_est - depth_gt).abs())
