import os
import random
import torch
import numpy as np
import torch.nn.functional as F


def seed_everything(seed=None):
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    try:
        if seed is None:
            seed = os.environ.get("PL_GLOBAL_SEED", random.randint(min_seed_value, max_seed_value))
        seed = int(seed)
    except (TypeError, ValueError):
        seed = random.randint(min_seed_value, max_seed_value)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f'training seed is {seed}')


def get_class_vector(label_slice, num_classes):
    unique_labels = torch.unique(label_slice)
    unique_labels = unique_labels[unique_labels != 0]

    gt_lbl_torch = torch.zeros(num_classes, dtype=torch.float)
    for cls in unique_labels:
        gt_lbl_torch[int(cls) - 1] = 1.0

    return gt_lbl_torch


def soft_dice(pred, target, epsilon=1e-6):
    pred_probs = F.softmax(pred, dim=1)
    target_onehot = F.one_hot(target, num_classes=pred.shape[1])
    target_onehot = target_onehot.permute(0, 3, 1, 2).float()
    assert pred_probs.shape == target_onehot.shape

    dims = (0, 2, 3)
    intersection = torch.sum(pred_probs * target_onehot, dims)
    denominator = torch.sum(pred_probs + target_onehot, dims)

    dice_per_class = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return dice_per_class.mean()