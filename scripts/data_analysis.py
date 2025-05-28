#!/usr/bin/env python3
"""
种子数据集分析脚本
分析训练数据和竞品种子数据的结构、分布和特征
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from collections import defaultdict
import pandas as pd

def analyze_image_properties(image_path):
    """分析单张图像的属性"""
    try:
        img = cv2.imread(str(image_path))
        if img is None:
            return None
        
        height, width, channels = img.shape
        file_size = os.path.getsize(image_path) / (1024 * 1024)  # MB
        
        # 计算图像统计信息
        mean_bgr = np.mean(img, axis=(0, 1))
        std_bgr = np.std(img, axis=(0, 1))
        
        return {
            'path': str(image_path),
            'width': width,
            'height': height,
            'channels': channels,
            'file_size_mb': round(file_size, 2),
            'mean_b': round(mean_bgr[0], 2),
            'mean_g': round(mean_bgr[1], 2),
            'mean_r': round(mean_bgr[2], 2),
            'std_b': round(std_bgr[0], 2),
            'std_g': round(std_bgr[1], 2),
            'std_r': round(std_bgr[2], 2),
        }
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def analyze_dataset():
    """分析整个数据集"""
    project_root = Path(".")
    
    # 分析训练数据
    train_images_dir = project_root / "train" / "images"
    train_mask_dir = project_root / "train" / "mask"
    
    # 分析竞品种子数据
    ood_dir = project_root / "竞品种子"
    
    results = {
        'train_data': {
            'images': [],
            'masks': [],
            'total_images': 0,
            'total_masks': 0
        },
        'ood_data': {
            'categories': {},
            'total_images': 0
        },
        'summary': {}
    }
    
    print("=== 分析训练数据 ===")
    
    # 分析训练图像
    if train_images_dir.exists():
        image_files = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.JPG"))
        results['train_data']['total_images'] = len(image_files)
        
        print(f"找到 {len(image_files)} 张训练图像")
        
        for i, img_path in enumerate(image_files[:10]):  # 分析前10张作为样本
            props = analyze_image_properties(img_path)
            if props:
                results['train_data']['images'].append(props)
            
            if i % 5 == 0:
                print(f"已分析 {i+1}/{min(10, len(image_files))} 张图像...")
    
    # 分析训练mask
    if train_mask_dir.exists():
        mask_files = list(train_mask_dir.glob("*.jpg")) + list(train_mask_dir.glob("*.JPG")) + \
                    list(train_mask_dir.glob("*.png")) + list(train_mask_dir.glob("*.PNG"))
        results['train_data']['total_masks'] = len(mask_files)
        print(f"找到 {len(mask_files)} 张mask图像")
    
    print("\n=== 分析竞品种子数据（OOD数据）===")
    
    # 分析竞品种子数据
    if ood_dir.exists():
        for category_dir in ood_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                image_files = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.JPG"))
                
                results['ood_data']['categories'][category_name] = {
                    'count': len(image_files),
                    'images': []
                }
                results['ood_data']['total_images'] += len(image_files)
                
                print(f"{category_name}: {len(image_files)} 张图像")
                
                # 分析每个类别的前几张图像
                for img_path in image_files[:3]:
                    props = analyze_image_properties(img_path)
                    if props:
                        results['ood_data']['categories'][category_name]['images'].append(props)
    
    # 生成总结
    results['summary'] = {
        'total_train_images': results['train_data']['total_images'],
        'total_train_masks': results['train_data']['total_masks'],
        'total_ood_images': results['ood_data']['total_images'],
        'ood_categories': len(results['ood_data']['categories']),
        'ood_category_names': list(results['ood_data']['categories'].keys())
    }
    
    return results

def create_data_summary_report(results):
    """创建数据摘要报告"""
    print("\n" + "="*60)
    print("种子数据集分析报告")
    print("="*60)
    
    print(f"\n📊 数据集概览:")
    print(f"  训练图像: {results['summary']['total_train_images']} 张")
    print(f"  训练mask: {results['summary']['total_train_masks']} 张")
    print(f"  OOD图像: {results['summary']['total_ood_images']} 张")
    print(f"  OOD类别: {results['summary']['ood_categories']} 种")
    
    print(f"\n🏷️  OOD类别分布:")
    for category, data in results['ood_data']['categories'].items():
        print(f"  {category}: {data['count']} 张")
    
    # 分析图像属性
    if results['train_data']['images']:
        train_images = results['train_data']['images']
        
        print(f"\n📐 训练图像属性分析 (基于 {len(train_images)} 张样本):")
        
        widths = [img['width'] for img in train_images]
        heights = [img['height'] for img in train_images]
        file_sizes = [img['file_size_mb'] for img in train_images]
        
        print(f"  图像尺寸:")
        print(f"    宽度: {min(widths)} - {max(widths)} (平均: {np.mean(widths):.0f})")
        print(f"    高度: {min(heights)} - {max(heights)} (平均: {np.mean(heights):.0f})")
        print(f"  文件大小: {min(file_sizes):.1f} - {max(file_sizes):.1f} MB (平均: {np.mean(file_sizes):.1f} MB)")
        
        # 颜色统计
        mean_r = np.mean([img['mean_r'] for img in train_images])
        mean_g = np.mean([img['mean_g'] for img in train_images])
        mean_b = np.mean([img['mean_b'] for img in train_images])
        
        print(f"  平均颜色值 (BGR): ({mean_b:.1f}, {mean_g:.1f}, {mean_r:.1f})")
    
    print(f"\n💡 数据集特点:")
    print(f"  - 这是一个种子分类和OOD检测数据集")
    print(f"  - 训练数据包含10种已知种子类型（需要进一步分类标注）")
    print(f"  - OOD数据包含5种外来种子类型，用于测试异常检测")
    print(f"  - 图像质量较高，文件大小约4MB左右")
    
    print(f"\n⚠️  注意事项:")
    print(f"  - 训练数据需要按种子类型进行分类标注")
    print(f"  - 当前训练图像都在同一个文件夹中，需要创建类别子文件夹")
    print(f"  - 建议将数据集划分为训练集、验证集和测试集")
    print(f"  - OOD数据可以直接用于异常检测测试")

def save_results(results, output_path="scripts/dataset_analysis.json"):
    """保存分析结果到JSON文件"""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n💾 分析结果已保存到: {output_path}")

def main():
    """主函数"""
    print("开始分析种子数据集...")
    
    # 分析数据集
    results = analyze_dataset()
    
    # 创建报告
    create_data_summary_report(results)
    
    # 保存结果
    save_results(results)
    
    print(f"\n✅ 数据集分析完成！")
    print(f"\n🚀 下一步建议:")
    print(f"  1. 对训练图像进行类别标注（创建10个类别文件夹）")
    print(f"  2. 实现数据加载器和预处理管道")
    print(f"  3. 设计数据增强策略")
    print(f"  4. 划分训练、验证、测试集")

if __name__ == "__main__":
    main() 