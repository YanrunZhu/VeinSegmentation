"""
为 Dataset230_purunRUS35 创建5折交叉验证划分
由于只有34张图像，每折约6-7张验证，27-28张训练
"""
import json
import os
from pathlib import Path

# 数据集路径
dataset_name = "Dataset230_purunRUS35"
nnUNet_preprocessed_path = "/home/zyr/nnUNet/nnUNet-wh/DATASET/nnUNet_preprocessed"
splits_file_path = os.path.join(nnUNet_preprocessed_path, dataset_name, "splits_final.json")

# 获取所有训练案例名称
nnUNet_raw_path = "/home/zyr/nnUNet/nnUNet-wh/DATASET/nnUNet_raw"
images_tr_dir = os.path.join(nnUNet_raw_path, dataset_name, "imagesTr")

# 读取所有图像文件名
image_files = [f for f in os.listdir(images_tr_dir) if f.endswith('.png')]
# 提取案例名称（去掉 _0000.png 后缀）
case_names = sorted([f.replace('_0000.png', '') for f in image_files if f.endswith('_0000.png')])

print(f"找到 {len(case_names)} 个训练案例")
print(f"案例名称示例: {case_names[:5]}")

# 创建5折交叉验证划分
num_folds = 5
num_cases = len(case_names)
cases_per_fold = num_cases // num_folds
remainder = num_cases % num_folds

splits = []
for fold_id in range(num_folds):
    # 计算当前fold的验证集范围
    val_start = fold_id * cases_per_fold
    val_end = val_start + cases_per_fold
    if fold_id < remainder:
        val_start += fold_id
        val_end += fold_id + 1
    else:
        val_start += remainder
        val_end += remainder
    
    # 提取验证集
    val_cases = case_names[val_start:val_end]
    # 训练集是其余所有案例
    train_cases = [c for c in case_names if c not in val_cases]
    
    splits.append({
        "train": train_cases,
        "val": val_cases
    })
    
    print(f"\nFold {fold_id}:")
    print(f"  训练集: {len(train_cases)} 张")
    print(f"  验证集: {len(val_cases)} 张")
    print(f"  验证集案例: {val_cases}")

# 确保输出目录存在
os.makedirs(os.path.dirname(splits_file_path), exist_ok=True)

# 保存划分文件
with open(splits_file_path, 'w') as f:
    json.dump(splits, f, indent=4)

print(f"\n✅ 5折交叉验证划分已保存到: {splits_file_path}")


