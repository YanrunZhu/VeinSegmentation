import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# 路径设置 - Dataset230_purunRUS35
image_dir = '/home/zyr/nnUNet/nnUNet-wh/DATASET/nnUNet_raw/Dataset230_purunRUS35/imagesTr'
label_dir = '/home/zyr/nnUNet/nnUNet-wh/DATASET/nnUNet_raw/Dataset230_purunRUS35/labelsTr'
pred_base_dir = '/home/zyr/nnUNet/nnUNet-wh/DATASET/nnUNet_trained_models/Dataset230_purunRUS35/nnUNetTrainerDA5__nnUNetPlans__2d'
output_base_dir = '/home/zyr/nnUNet/visual_all_folds_DA5_Dataset230'

def calculate_dice(y_true, y_pred):
    """
    计算 Dice 系数
    """
    intersection = np.sum(y_true * y_pred)
    return (2. * intersection) / (np.sum(y_true) + np.sum(y_pred) + 1e-8)

def visualize_and_save(image, label, pred, case_name, save_dir):
    """
    可视化并保存：将原始图像、真实标签和预测结果叠加显示
    - 原始图像：灰度背景
    - 真实标签：绿色半透明叠加
    - 预测结果：红色半透明叠加
    """
    dice_score = calculate_dice(label, pred)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(image, cmap='gray')

    # 创建 mask
    label_mask = np.zeros((*label.shape, 3))
    label_mask[label == 1] = [0, 1, 0]  # green

    pred_mask = np.zeros((*pred.shape, 3))
    pred_mask[pred == 1] = [1, 0, 0]    # red

    ax.imshow(label_mask, alpha=0.4)
    ax.imshow(pred_mask, alpha=0.4)

    ax.set_title(f'Id: {case_name}\nDice: {dice_score:.4f}', fontsize=10)
    ax.axis('off')

    # 图例
    legend_elements = [
        Patch(facecolor='green', alpha=0.4, label='Ground Truth'),
        Patch(facecolor='red', alpha=0.4, label='Prediction')
    ]
    ax.legend(handles=legend_elements, loc='lower left')

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{case_name}_vis.png")
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"✅ 已保存: {save_path}")

# 遍历每一个 fold
for fold_id in range(5):
    print(f"\n🔍 正在处理 fold_{fold_id} ...")
    pred_dir = os.path.join(pred_base_dir, f'fold_{fold_id}', 'validation')
    output_dir = os.path.join(output_base_dir, f'fold_{fold_id}')

    # 检查预测路径是否存在并非空
    if not os.path.exists(pred_dir):
        print(f"❌ fold_{fold_id} 路径不存在: {pred_dir}")
        print(f"   提示: 请确保已完成 fold_{fold_id} 的训练和验证")
        continue

    pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.png')]
    if len(pred_files) == 0:
        print(f"⚠️ fold_{fold_id} 路径存在但为空: {pred_dir}")
        continue

    # 找到三类图像中都有的图名
    # 图像文件名格式: benign0001_0000.png
    image_names = {f.replace('_0000.png', '').replace('.png', '') for f in os.listdir(image_dir) if f.endswith('.png')}
    # 标签文件名格式: benign0001.png
    label_names = {f.replace('.png', '') for f in os.listdir(label_dir) if f.endswith('.png')}
    # 预测文件名格式: benign0001.png
    pred_names = {f.replace('.png', '') for f in pred_files}

    common_names = sorted(list(image_names & label_names & pred_names))
    print(f"📸 fold_{fold_id} 中共找到 {len(common_names)} 张图像可视化")

    if len(common_names) == 0:
        print(f"⚠️ fold_{fold_id} 没有找到可以处理的图像（可能文件名不一致或预测不全）")
        print(f"   图像文件: {len(image_names)} 个")
        print(f"   标签文件: {len(label_names)} 个")
        print(f"   预测文件: {len(pred_names)} 个")
        continue

    for name in common_names:
        image_path = os.path.join(image_dir, name + '_0000.png')
        label_path = os.path.join(label_dir, name + '.png')
        pred_path = os.path.join(pred_dir, name + '.png')

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        if image is None or label is None or pred is None:
            print(f"⚠️ 跳过无法读取的图像: {name}")
            continue

        # 二值化
        label = (label > 0).astype(np.uint8)
        pred = (pred > 0).astype(np.uint8)

        visualize_and_save(image, label, pred, name, output_dir)

print("\n🎉 所有 folds 的图像处理完成！")
print(f"📁 可视化结果保存在: {output_base_dir}")

