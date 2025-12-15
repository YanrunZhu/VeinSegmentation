# 文件路径: /home/zyr/nnUNet/nnunetv2/training/nnUNetTrainer/CustomDiceTrainer.py
import torch
import numpy as np
import warnings # 导入 warnings 模块

# 从同一目录下的 nnUNetTrainer.py 文件中导入 nnUNetTrainer 类
from .nnUNetTrainer import nnUNetTrainer # <--- 使用相对导入，指向同目录的 nnUNetTrainer.py

# 从同一目录下的 custom_losses.py 文件中导入 WeightedSoftDiceLoss 类
from .custom_losses import WeightedSoftDiceLoss # <--- 使用相对导入

# 从 nnU-Net 的标准路径导入其他必要的损失组件
# (这些路径假设您的 PYTHONPATH 设置正确，能够找到 nnunetv2 包)
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import MemoryEfficientSoftDiceLoss # 用于基于区域的默认损失


class CustomDiceTrainer(nnUNetTrainer): # <--- 继承自 nnUNetTrainer
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device('cuda')):
        # 调用父类 (nnUNetTrainer) 的构造函数
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.print_to_log_file("Initializing CustomDiceTrainer, inheriting from nnUNetTrainer, with WeightedSoftDiceLoss!")

    def _build_loss(self):
        self.print_to_log_file("CustomDiceTrainer: Building custom weighted loss function.")
        # --- 根据您的数据集信息配置权重 ---
        # "labels": {"background": 0, ..., "vessels": 4, ...}
        num_total_classes = 9  # 从0 (背景) 到 8 (脾脏)
        vessels_label_index = 4  # "vessels" 是标签 4
        boost_factor = 5.0 # 对 "vessels" 的提升因子

        # Dice 损失的权重配置
        # 参考您 nnUNetTrainer.py 中 _build_loss 方法对非区域情况的 do_bg 设置
        # 通常，对于非区域（基于类别）的分割，DC_and_CE_loss 的 do_bg=False
        DO_BG_FOR_DICE_LOSS = False

        if DO_BG_FOR_DICE_LOSS:
            dice_class_weights = torch.ones(num_total_classes, device=self.device)
            dice_class_weights[vessels_label_index] = boost_factor
        else:
            # 如果 do_bg=False，Dice 权重应针对前景类。
            num_foreground_classes = num_total_classes - 1
            dice_class_weights = torch.ones(num_foreground_classes, device=self.device)
            if vessels_label_index > 0: # 确保 "vessels" 不是背景标签 (索引0)
                 vessels_foreground_index = vessels_label_index - 1 # 计算在前景类别中的索引
                 dice_class_weights[vessels_foreground_index] = boost_factor
            else:
                # 如果 vessels_label_index 是0（背景），并且 DO_BG_FOR_DICE_LOSS 是 False，
                # 那么为前景类准备的 dice_class_weights 不会包含背景，这通常是期望的行为。
                # 但如果您的目标是背景，这里的逻辑需要调整。鉴于 vessels_label_index=4，这不是问题。
                pass
        
        self.print_to_log_file(f"Dice class weights (for WeightedSoftDiceLoss, do_bg={DO_BG_FOR_DICE_LOSS}): {dice_class_weights.tolist()}")

        # CE 损失的权重配置 (通常应用于所有类别，包括背景)
        ce_class_weights = torch.ones(num_total_classes, device=self.device)
        ce_class_weights[vessels_label_index] = boost_factor
        self.print_to_log_file(f"CE class weights: {ce_class_weights.tolist()}")

        # --- 构建损失 ---
        # 检查是否是基于区域的训练 (此逻辑应与您的 nnUNetTrainer.py 中的 _build_loss 方法一致)
        if self.label_manager.has_regions:
            self.print_to_log_file("Region-based training detected. Applying nnU-Net's default BCE-Dice for regions. WeightedSoftDiceLoss will NOT be used here.")
            # 对于基于区域的训练，nnUNetTrainer.py 中通常使用 DC_and_BCE_loss 和 MemoryEfficientSoftDiceLoss
            # 在这种情况下，我们的自定义加权可能不直接适用或需要更复杂的逻辑来处理区域的独立性
            # 因此，这里我们选择使用 nnUNetTrainer.py 中对区域的默认处理方式，不应用自定义的 WeightedSoftDiceLoss
            loss = DC_and_BCE_loss({}, # bce_kwargs
                                   {'batch_dice': self.configuration_manager.batch_dice,
                                    'do_bg': True, # 对于BCE，通常包含所有“区域”作为前景
                                    'smooth': 1e-5, 
                                    'ddp': self.is_ddp}, # soft_dice_kwargs
                                   use_ignore_label=self.label_manager.ignore_label is not None,
                                   dice_class=MemoryEfficientSoftDiceLoss) # 使用 nnU-Net 默认的 Dice 实现
            warnings.warn("CustomDiceTrainer: Region-based training active. Falling back to nnU-Net's default DC_and_BCE_loss with MemoryEfficientSoftDiceLoss for regions. WeightedSoftDiceLoss is NOT used in this branch.")
        else:
            # 标准的基于类别的分割 (非区域)
            self.print_to_log_file("Standard (non-region) segmentation training. Applying WeightedSoftDiceLoss.")
            loss = DC_and_CE_loss(
                soft_dice_kwargs={
                    'batch_dice': self.configuration_manager.batch_dice,
                    'smooth': 1e-5,
                    'do_bg': DO_BG_FOR_DICE_LOSS, # 根据上面定义
                    'ddp': self.is_ddp,
                    'class_weights': dice_class_weights # <--- 在这里传递Dice的类别权重
                },
                ce_kwargs={'weight': ce_class_weights}, # <--- 在这里传递CE的类别权重
                weight_ce=1, weight_dice=1,
                ignore_label=self.label_manager.ignore_label,
                dice_class=WeightedSoftDiceLoss # <--- 在这里指定使用我们的加权Dice损失
            )

        # 编译Dice部分 (如果启用了编译，并且损失对象有 'dc' 属性)
        # 这个 _do_i_compile 方法应该从父类 nnUNetTrainer 继承而来
        if self._do_i_compile():
            if hasattr(loss, 'dc') and loss.dc is not None:
                 loss.dc = torch.compile(loss.dc)

        # 应用深度监督 (如果启用)
        # self.enable_deep_supervision 和 self._get_deep_supervision_scales 应该从父类继承
        if self.enable_deep_supervision:
            deep_supervision_scales = self._get_deep_supervision_scales()
            weights = np.array([1 / (2 ** i) for i in range(len(deep_supervision_scales))])
            # 处理DDP和编译交互的权重逻辑，与 nnUNetTrainer.py 中的逻辑保持一致
            if self.is_ddp and not self._do_i_compile() and len(weights) > 0:
                weights[-1] = 1e-6
            elif len(weights) > 0:
                weights[-1] = 0

            if len(weights) > 0 : # 避免在没有深度监督尺度时除以零
                weights = weights / weights.sum()
            loss = DeepSupervisionWrapper(loss, weights)
        return loss