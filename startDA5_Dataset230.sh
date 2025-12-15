#!/bin/bash

# Dataset ID
dataset_id=230
# Configuration
configuration="2d"
# Custom trainer
trainer="nnUNetTrainerDA5"

# 清理 LD_LIBRARY_PATH 以避免 cuDNN 版本冲突
# PyTorch 自带 cuDNN，不需要系统路径中的旧版本
if [ -n "$LD_LIBRARY_PATH" ]; then
    # 移除包含 cudnn 和 cuda 的路径
    export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v -i cudnn | grep -v -i cuda | tr '\n' ':' | sed 's/:$//')
fi

# Loop through folds 0, 1, 2, 3, 4
for fold_num in 0 1 2 3 4
do
  echo "Starting training for fold $fold_num on GPU 0..."
  # Set CUDA_VISIBLE_DEVICES to 0 for the nnUNetv2_train command
  # 禁用 torch.compile 以避免 cuDNN 相关问题
  CUDA_VISIBLE_DEVICES=0 NNUNET_DO_NOT_COMPILE=1 nnUNetv2_train $dataset_id $configuration $fold_num -tr $trainer
  echo "Completed training for fold $fold_num."
done

echo "All 5-fold cross-validation training finished."

