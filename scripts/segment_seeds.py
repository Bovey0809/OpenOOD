#!/usr/bin/env python3
"""
种子分割脚本
使用mask文件分割出单个种子，创建用于训练的数据集
mask中0表示背景，其他数值表示不同种子的类别
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import os
from tqdm import tqdm

def load_image_and_mask(image_path, mask_path):
    """加载图像和对应的mask"""
    # 加载原图
    image = cv2.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 加载mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    
    return image, mask

def find_seed_contours_by_class(mask, min_area=500):
    """在mask中找到不同类别种子的轮廓"""
    # 检查mask的唯一值
    unique_values = np.unique(mask)
    print(f"  Mask unique values: {unique_values}")
    
    # 0是背景，其他值是种子类别
    seed_classes = [val for val in unique_values if val != 0]
    print(f"  Found seed classes: {seed_classes}")
    
    class_contours = {}
    
    # 为每个种子类别找轮廓
    for seed_class in seed_classes:
        # 创建该类别的二值mask
        class_mask = np.zeros_like(mask)
        class_mask[mask == seed_class] = 255
        
        # 形态学操作，去除噪声
        kernel = np.ones((5,5), np.uint8)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_CLOSE, kernel)
        class_mask = cv2.morphologyEx(class_mask, cv2.MORPH_OPEN, kernel)
        
        # 找轮廓
        contours, _ = cv2.findContours(class_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 过滤小的轮廓
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area:
                valid_contours.append(contour)
        
        if valid_contours:
            class_contours[seed_class] = valid_contours
            print(f"    Class {seed_class}: {len(valid_contours)} valid contours")
    
    return class_contours

def extract_seed_from_contour(image, mask, contour, seed_class, padding=20):
    """从轮廓中提取种子图像"""
    # 获取边界框
    x, y, w, h = cv2.boundingRect(contour)
    
    # 添加padding
    x_start = max(0, x - padding)
    y_start = max(0, y - padding)
    x_end = min(image.shape[1], x + w + padding)
    y_end = min(image.shape[0], y + h + padding)
    
    # 提取种子区域
    seed_image = image[y_start:y_end, x_start:x_end]
    seed_mask_region = mask[y_start:y_end, x_start:x_end]
    
    # 创建该种子的精确mask
    seed_mask = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)
    seed_mask[seed_mask_region == seed_class] = 255
    
    # 应用mask（设置背景为白色）
    seed_image_masked = seed_image.copy()
    seed_image_masked[seed_mask == 0] = [255, 255, 255]  # 白色背景
    
    return seed_image_masked, seed_mask

def process_image_pair(image_path, mask_path, output_dir, image_id):
    """处理一对图像和mask"""
    try:
        # 加载图像和mask
        image, mask = load_image_and_mask(image_path, mask_path)
        
        # 找到不同类别种子的轮廓
        class_contours = find_seed_contours_by_class(mask, min_area=500)
        
        extracted_seeds = []
        seed_counter = 0
        
        # 处理每个类别的种子
        for seed_class, contours in class_contours.items():
            for i, contour in enumerate(contours):
                # 提取种子
                seed_image, seed_mask = extract_seed_from_contour(image, mask, contour, seed_class)
                
                # 检查种子大小
                if seed_image.shape[0] < 50 or seed_image.shape[1] < 50:
                    continue  # 跳过太小的种子
                
                # 保存种子图像
                seed_filename = f"{image_id}_class{seed_class}_seed_{seed_counter:02d}.jpg"
                seed_path = output_dir / seed_filename
                
                # 转换回BGR用于保存
                seed_bgr = cv2.cvtColor(seed_image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(seed_path), seed_bgr)
                
                # 记录种子信息
                seed_info = {
                    'filename': seed_filename,
                    'path': str(seed_path),
                    'original_image': str(image_path),
                    'original_mask': str(mask_path),
                    'seed_class': int(seed_class),  # 真实的种子类别
                    'contour_area': cv2.contourArea(contour),
                    'bbox': cv2.boundingRect(contour),
                    'width': seed_image.shape[1],
                    'height': seed_image.shape[0]
                }
                
                extracted_seeds.append(seed_info)
                seed_counter += 1
        
        return extracted_seeds
        
    except Exception as e:
        print(f"处理 {image_path} 时出错: {e}")
        return []

def segment_all_seeds():
    """分割所有种子"""
    project_root = Path(".")
    
    # 输入路径
    train_images_dir = project_root / "train" / "images"
    train_masks_dir = project_root / "train" / "mask"
    
    # 输出路径
    output_dir = project_root / "datasets" / "seeds" / "segmented"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有图像文件
    image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.JPG"))
    
    print(f"找到 {len(image_files)} 张训练图像")
    
    all_seeds_info = []
    total_seeds = 0
    class_counts = defaultdict(int)
    
    # 处理每张图像
    for image_path in tqdm(image_files, desc="分割种子"):
        # 找到对应的mask文件
        image_stem = image_path.stem
        
        # 尝试不同的mask文件名格式
        possible_mask_names = [
            f"{image_stem}.png",
            f"{image_stem}.PNG", 
            f"{image_stem}.jpg",
            f"{image_stem}.JPG"
        ]
        
        mask_path = None
        for mask_name in possible_mask_names:
            potential_mask = train_masks_dir / mask_name
            if potential_mask.exists():
                mask_path = potential_mask
                break
        
        if mask_path is None:
            print(f"⚠️  未找到 {image_path.name} 对应的mask文件")
            continue
        
        # 处理图像对
        image_id = image_path.stem
        seeds_info = process_image_pair(image_path, mask_path, output_dir, image_id)
        
        # 统计类别
        for seed_info in seeds_info:
            class_counts[seed_info['seed_class']] += 1
        
        all_seeds_info.extend(seeds_info)
        total_seeds += len(seeds_info)
        
        if len(seeds_info) > 0:
            print(f"  {image_path.name}: 提取了 {len(seeds_info)} 个种子")
    
    # 保存分割信息
    segmentation_info = {
        'total_original_images': len(image_files),
        'total_extracted_seeds': total_seeds,
        'seeds_per_image': total_seeds / len(image_files) if len(image_files) > 0 else 0,
        'class_distribution': dict(class_counts),
        'unique_classes': list(class_counts.keys()),
        'num_classes': len(class_counts),
        'output_directory': str(output_dir),
        'seeds_info': all_seeds_info
    }
    
    info_path = output_dir / "segmentation_info.json"
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(segmentation_info, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ 种子分割完成！")
    print(f"总共提取了 {total_seeds} 个种子")
    print(f"平均每张图像: {total_seeds / len(image_files):.1f} 个种子")
    print(f"发现 {len(class_counts)} 个种子类别:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  类别 {class_id}: {count} 个种子")
    print(f"分割信息已保存到: {info_path}")
    
    return segmentation_info

def visualize_segmentation_results(num_samples=12):
    """可视化分割结果，按类别显示"""
    output_dir = Path("datasets/seeds/segmented")
    info_path = output_dir / "segmentation_info.json"
    
    if not info_path.exists():
        print("❌ 分割信息文件不存在，请先运行分割")
        return
    
    with open(info_path, 'r', encoding='utf-8') as f:
        segmentation_info = json.load(f)
    
    seeds_info = segmentation_info['seeds_info']
    
    if len(seeds_info) == 0:
        print("❌ 没有找到分割的种子")
        return
    
    # 按类别组织种子
    seeds_by_class = defaultdict(list)
    for seed_info in seeds_info:
        seeds_by_class[seed_info['seed_class']].append(seed_info)
    
    # 为每个类别选择样本
    sample_seeds = []
    classes = sorted(seeds_by_class.keys())
    samples_per_class = max(1, num_samples // len(classes))
    
    for class_id in classes:
        class_seeds = seeds_by_class[class_id]
        import random
        selected = random.sample(class_seeds, min(samples_per_class, len(class_seeds)))
        sample_seeds.extend(selected)
    
    # 限制总样本数
    sample_seeds = sample_seeds[:num_samples]
    
    # 创建可视化
    rows = 3
    cols = 4
    fig, axes = plt.subplots(rows, cols, figsize=(16, 12))
    fig.suptitle('分割的种子样本（按类别）', fontsize=16)
    
    for i, seed_info in enumerate(sample_seeds):
        if i >= rows * cols:
            break
            
        row = i // cols
        col = i % cols
        
        # 加载种子图像
        seed_path = Path(seed_info['path'])
        if seed_path.exists():
            seed_image = cv2.imread(str(seed_path))
            seed_image = cv2.cvtColor(seed_image, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(seed_image)
            axes[row, col].set_title(f"类别 {seed_info['seed_class']}\n{seed_info['filename']}", 
                                   fontsize=10)
            axes[row, col].axis('off')
        else:
            axes[row, col].text(0.5, 0.5, '图像不存在', ha='center', va='center')
            axes[row, col].axis('off')
    
    # 隐藏多余的子图
    for i in range(len(sample_seeds), rows * cols):
        row = i // cols
        col = i % cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    
    # 保存可视化结果
    viz_path = output_dir / "segmentation_samples_by_class.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"📊 分割样本可视化已保存到: {viz_path}")
    
    # 显示类别分布
    print(f"\n📈 种子类别分布:")
    for class_id in sorted(seeds_by_class.keys()):
        count = len(seeds_by_class[class_id])
        print(f"  类别 {class_id}: {count} 个种子")
    
    try:
        plt.show()
    except:
        print("⚠️  无法显示图像，但已保存到文件")

def main():
    """主函数"""
    print("🔪 开始种子分割处理...")
    print("📝 注意：mask中0表示背景，其他数值表示种子类别")
    
    # 执行分割
    segmentation_info = segment_all_seeds()
    
    # 可视化结果
    print("\n📊 生成可视化结果...")
    visualize_segmentation_results()
    
    print(f"\n🎉 种子分割处理完成！")
    print(f"发现了 {segmentation_info['num_classes']} 个种子类别")
    print(f"下一步: 使用真实类别标签训练分类模型")

if __name__ == "__main__":
    main() 