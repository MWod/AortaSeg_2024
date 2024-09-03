### Ecosystem Imports ###
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "."))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from typing import Union
from enum import Enum

### External Imports ###
import torch as tc
import torch.nn.functional as F
from monai import losses

### Internal Imports ###
import hausdorff
import cl_dice

########################


### Volumetric Losses ###

def dice_loss(prediction : tc.Tensor, target : tc.Tensor) -> tc.Tensor:
    """
    Dice as PyTorch cost function.
    """
    smooth = 1
    prediction = prediction.contiguous().view(-1)
    target = target.contiguous().view(-1)
    intersection = tc.sum(prediction * target)
    return 1 - ((2 * intersection + smooth) / (prediction.sum() + target.sum() + smooth))

def dice_loss_multichannel(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Dice loss for multichannel masks (equally averaged)
    """
    try:
        sigmoid = kwargs['sigmoid']
    except:
        sigmoid = True
        
    no_channels = prediction.size(1)
    for i in range(no_channels):
        if i == 0:
            if sigmoid:
                loss = dice_loss(tc.sigmoid(prediction[:, i, :, :, :]), target[:, i, :, :, :])
            else:
                loss = dice_loss(prediction[:, i, :, :, :], target[:, i, :, :, :])
        else:
            if sigmoid:
                loss += dice_loss(tc.sigmoid(prediction[:, i, :, :, :]), target[:, i, :, :, :])
            else:
                loss += dice_loss(prediction[:, i, :, :, :], target[:, i, :, :, :])
    loss = loss / no_channels
    return loss

def hausdorff_loss(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    """
    return hausdorff.HDLoss(**kwargs)(prediction, target)


########################

### MONAI Volumetric Losses ###

def dice_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Dice loss based on MONAI implementation.
    """
    return losses.DiceLoss(reduction='mean', **kwargs)(prediction, target)

def dice_ce_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Cross Entropy losses based on MONAI implementation.
    """
    return losses.DiceCELoss(reduction='mean', **kwargs)(prediction, target)

def dice_ce_loss_monai_2(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Cross Entropy losses based on MONAI implementation.
    """
    return losses.DiceCELoss(reduction='mean', sigmoid=True, **kwargs)(prediction, target)

def dice_focal_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Focal losses based on MONAI implementation.
    """
    return losses.DiceFocalLoss(reduction='mean', **kwargs)(prediction, target)

def generalized_dice_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Generalized Dice loss based on MONAI implementation.
    """
    return losses.GeneralizedDiceLoss(reduction='mean', **kwargs)(prediction, target)

def generalized_dice_focal_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Generalized Dice and Focal losses based on MONAI implementation.
    """
    return losses.GeneralizedDiceFocalLoss(reduction='mean', **kwargs)(prediction, target)

def tversky_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Focal losses based on MONAI implementation.
    """
    return losses.TverskyLoss(reduction='mean', **kwargs)(prediction, target)

def dice_focal_cldice_loss_monai(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Focal losses based on MONAI implementation.
    """
    dice_focal = losses.DiceFocalLoss(reduction='mean', **kwargs)(prediction, target)
    if kwargs['sigmoid']:
        cldice =  cl_dice.soft_dice_cldice()(tc.sigmoid(prediction), target)
    else:
        cldice =  cl_dice.soft_dice_cldice()(prediction, target)
    return dice_focal + cldice

def dice_focal_cldice_loss_monai_v2(prediction : tc.Tensor, target : tc.Tensor, **kwargs) -> tc.Tensor:
    """
    Averaged Dice and Focal losses based on MONAI implementation.
    """
    dice_focal = losses.DiceFocalLoss(reduction='mean', **kwargs)(prediction, target)
    if kwargs['sigmoid']:
        cldice =  cl_dice.soft_dice_cldice_v2()(tc.sigmoid(prediction), target)
    else:
        cldice =  cl_dice.soft_dice_cldice_v2()(prediction, target)
    return dice_focal + cldice



########################