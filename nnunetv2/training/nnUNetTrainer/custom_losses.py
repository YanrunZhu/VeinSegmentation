# /home/zyr/nnUNet/nnunetv2/training/nnUNetTrainer/custom_losses.py
import torch
from torch import nn
from typing import Callable, Union, List, Tuple

# 假设这些 nnU-Net 的工具函数可以被正确导入
# 这些导入路径依赖于您的PYTHONPATH是否正确设置，使得nnunetv2成为一个可识别的顶级包
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn
from nnunetv2.utilities.ddp_allgather import AllGatherGrad
# from nnunetv2.utilities.helpers import softmax_helper_dim1 # 通常由训练器传递

class WeightedSoftDiceLoss(nn.Module):
    def __init__(self, class_weights: torch.Tensor, apply_nonlin: Callable = None, batch_dice: bool = False,
                 do_bg: bool = True, smooth: float = 1., ddp: bool = False, clip_tp: float = None):
        super().__init__()
        self.class_weights = class_weights
        self.apply_nonlin = apply_nonlin
        self.batch_dice = batch_dice
        self.do_bg = do_bg
        self.smooth = smooth
        self.ddp = ddp
        self.clip_tp = clip_tp

    def forward(self, x_pred: torch.Tensor, y_true: torch.Tensor, loss_mask: torch.Tensor = None) -> torch.Tensor:
        shp_x = x_pred.shape

        if self.batch_dice:
            axes = [0] + list(range(2, len(shp_x)))
        else:
            axes = list(range(2, len(shp_x)))

        if self.apply_nonlin is not None:
            x_pred = self.apply_nonlin(x_pred)

        tp, fp, fn, _ = get_tp_fp_fn_tn(x_pred, y_true, axes, loss_mask, False)

        if self.ddp and self.batch_dice:
            tp = AllGatherGrad.apply(tp).sum(0)
            fp = AllGatherGrad.apply(fp).sum(0)
            fn = AllGatherGrad.apply(fn).sum(0)

        if self.clip_tp is not None:
            tp = torch.clip(tp, min=self.clip_tp, max=None)

        nominator = 2 * tp
        denominator = 2 * tp + fp + fn
        dc_per_class = (nominator + self.smooth) / (torch.clip(denominator + self.smooth, 1e-8))

        if self.class_weights.device != dc_per_class.device:
            self.class_weights = self.class_weights.to(dc_per_class.device)

        active_weights = self.class_weights
        if not self.do_bg:
            if dc_per_class.ndim == 1:
                dc_per_class = dc_per_class[1:]
                if len(self.class_weights) == shp_x[1]:
                    active_weights = self.class_weights[1:]
            elif dc_per_class.ndim == 2:
                dc_per_class = dc_per_class[:, 1:]
                if len(self.class_weights) == shp_x[1]:
                    active_weights = self.class_weights[1:].unsqueeze(0)
                else:
                    active_weights = self.class_weights.unsqueeze(0)
        elif dc_per_class.ndim == 2 and self.do_bg :
             active_weights = self.class_weights.unsqueeze(0)

        if dc_per_class.shape[-1] != active_weights.shape[-1]:
            raise ValueError(
                f"处理 do_bg 后的形状不匹配: "
                f"dc_per_class 有 {dc_per_class.shape[-1]} 个类别, "
                f"但 active_weights 有 {active_weights.shape[-1]} 个元素。 "
                f"do_bg={self.do_bg}, 初始类别数 (来自 x_pred)={shp_x[1]}, "
                f"初始 class_weights_len={len(self.class_weights)}"
            )

        weighted_dc = dc_per_class * active_weights
        dc = weighted_dc.mean()

        return -dc