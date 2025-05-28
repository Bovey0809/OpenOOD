#!/usr/bin/env python3
"""
种子数据集可视化脚本
展示训练数据和OOD数据的样本
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
import random

def load_and_display_samples():
    """加载并显示数据集样本"""
    
    # 加载数据集信息
    dataset_info_path = Path("datasets/seeds/dataset_info.json")
    if not dataset_info_path.exists():
        print("❌ 数据集信息文件不存在，请先运行 prepare_dataset.py")
        return
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    print("=== 种子数据集可视化 ===")
    print(f"训练集: {dataset_info['train_size']} 张")
    print(f"验证集: {dataset_info['val_size']} 张")
    print(f"OOD测试集: {dataset_info['ood_size']} 张")
    
    # 创建图像网格
    fig, axes = plt.subplots(3, 6, figsize=(18, 9))
    fig.suptitle('种子数据集样本展示', fontsize=16)
    
    # 显示训练样本
    train_paths = dataset_info['train_paths']
    train_labels = dataset_info['train_labels']
    
    print("\n=== 显示训练样本 ===")
    for i in range(6):
        if i < len(train_paths):
            img_path = Path(train_paths[i])
            label = train_labels[i]
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                axes[0, i].imshow(img)
                axes[0, i].set_title(f'训练样本\n类别: {label}', fontsize=10)
                axes[0, i].axis('off')
                
                print(f"  样本 {i+1}: {img_path.name} (类别: {label})")
            else:
                axes[0, i].text(0.5, 0.5, '图像不存在', ha='center', va='center')
                axes[0, i].axis('off')
    
    # 显示验证样本
    val_paths = dataset_info['val_paths']
    val_labels = dataset_info['val_labels']
    
    print("\n=== 显示验证样本 ===")
    for i in range(6):
        if i < len(val_paths):
            img_path = Path(val_paths[i])
            label = val_labels[i]
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                axes[1, i].imshow(img)
                axes[1, i].set_title(f'验证样本\n类别: {label}', fontsize=10)
                axes[1, i].axis('off')
                
                print(f"  样本 {i+1}: {img_path.name} (类别: {label})")
            else:
                axes[1, i].text(0.5, 0.5, '图像不存在', ha='center', va='center')
                axes[1, i].axis('off')
    
    # 显示OOD样本
    ood_paths = dataset_info['ood_paths']
    
    print("\n=== 显示OOD样本 ===")
    for i in range(6):
        if i < len(ood_paths):
            img_path = Path(ood_paths[i])
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (224, 224))
                
                # 从路径中提取OOD类别
                ood_category = img_path.parent.name
                
                axes[2, i].imshow(img)
                axes[2, i].set_title(f'OOD样本\n{ood_category}', fontsize=10)
                axes[2, i].axis('off')
                
                print(f"  样本 {i+1}: {img_path.name} ({ood_category})")
            else:
                axes[2, i].text(0.5, 0.5, '图像不存在', ha='center', va='center')
                axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # 保存图像
    output_path = Path("scripts/dataset_samples.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 样本图像已保存到: {output_path}")
    
    # 显示图像（如果在支持的环境中）
    try:
        plt.show()
    except:
        print("⚠️  无法显示图像，但已保存到文件")

def analyze_label_distribution():
    """分析标签分布"""
    dataset_info_path = Path("datasets/seeds/dataset_info.json")
    
    with open(dataset_info_path, 'r', encoding='utf-8') as f:
        dataset_info = json.load(f)
    
    # 统计训练集标签分布
    train_labels = dataset_info['train_labels']
    val_labels = dataset_info['val_labels']
    
    from collections import Counter
    
    train_counter = Counter(train_labels)
    val_counter = Counter(val_labels)
    
    print("\n=== 标签分布分析 ===")
    print("训练集标签分布:")
    for label in sorted(train_counter.keys()):
        print(f"  类别 {label}: {train_counter[label]} 张")
    
    print("\n验证集标签分布:")
    for label in sorted(val_counter.keys()):
        print(f"  类别 {label}: {val_counter[label]} 张")
    
    # 绘制分布图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # 训练集分布
    labels = sorted(train_counter.keys())
    counts = [train_counter[label] for label in labels]
    ax1.bar(labels, counts, color='skyblue', alpha=0.7)
    ax1.set_title('训练集标签分布')
    ax1.set_xlabel('类别')
    ax1.set_ylabel('图像数量')
    ax1.grid(True, alpha=0.3)
    
    # 验证集分布
    val_counts = [val_counter[label] for label in labels]
    ax2.bar(labels, val_counts, color='lightcoral', alpha=0.7)
    ax2.set_title('验证集标签分布')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('图像数量')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存分布图
    output_path = Path("scripts/label_distribution.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n💾 标签分布图已保存到: {output_path}")
    
    try:
        plt.show()
    except:
        print("⚠️  无法显示图像，但已保存到文件")

def main():
    """主函数"""
    print("开始可视化种子数据集...")
    
    # 显示样本
    load_and_display_samples()
    
    # 分析标签分布
    analyze_label_distribution()
    
    print("\n✅ 数据集可视化完成！")
    print("\n📊 总结:")
    print("  - 已生成数据集样本展示图")
    print("  - 已生成标签分布分析图")
    print("  - 当前使用伪标签，建议进行人工标注以提高准确性")

if __name__ == "__main__":
    main() 