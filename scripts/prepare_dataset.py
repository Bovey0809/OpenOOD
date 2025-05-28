#!/usr/bin/env python3
"""
种子数据集预处理脚本
准备用于训练的数据集结构
"""

import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
import json
import shutil
from sklearn.model_selection import train_test_split
import random

class SeedDataset(Dataset):
    """种子数据集类"""
    
    def __init__(self, image_paths, labels=None, transform=None, is_ood=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.is_ood = is_ood
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        if self.is_ood:
            # OOD数据，返回-1作为标签
            return image, -1
        else:
            # 正常数据
            label = self.labels[idx] if self.labels is not None else 0
            return image, label

def get_transforms(image_size=224, is_training=True):
    """获取数据变换"""
    if is_training:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform

def create_pseudo_labels(image_paths, num_classes=10):
    """
    为训练图像创建伪标签
    由于没有真实标签，我们基于文件名或其他特征创建伪标签
    """
    # 简单策略：基于文件名哈希分配标签
    labels = []
    for path in image_paths:
        # 使用文件名的哈希值来分配类别
        filename = Path(path).stem
        hash_value = hash(filename)
        label = hash_value % num_classes
        labels.append(label)
    
    return labels

def organize_dataset():
    """组织数据集结构"""
    project_root = Path(".")
    
    # 创建数据集目录
    dataset_dir = project_root / "datasets" / "seeds"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    
    # 训练数据路径
    train_images_dir = project_root / "train" / "images"
    ood_dir = project_root / "竞品种子"
    
    # 收集训练图像
    train_image_paths = []
    if train_images_dir.exists():
        train_image_paths = list(train_images_dir.glob("*.jpg")) + list(train_images_dir.glob("*.JPG"))
    
    print(f"找到 {len(train_image_paths)} 张训练图像")
    
    # 收集OOD图像
    ood_image_paths = []
    ood_categories = {}
    
    if ood_dir.exists():
        for category_dir in ood_dir.iterdir():
            if category_dir.is_dir():
                category_name = category_dir.name
                category_images = list(category_dir.glob("*.jpg")) + list(category_dir.glob("*.JPG"))
                ood_image_paths.extend(category_images)
                ood_categories[category_name] = len(category_images)
                print(f"OOD类别 {category_name}: {len(category_images)} 张图像")
    
    print(f"总共找到 {len(ood_image_paths)} 张OOD图像")
    
    # 创建伪标签（因为没有真实的种子类别标注）
    train_labels = create_pseudo_labels(train_image_paths, num_classes=10)
    
    # 划分训练集和验证集
    train_paths, val_paths, train_labels_split, val_labels_split = train_test_split(
        train_image_paths, train_labels, test_size=0.2, random_state=42, stratify=train_labels
    )
    
    print(f"训练集: {len(train_paths)} 张图像")
    print(f"验证集: {len(val_paths)} 张图像")
    
    # 保存数据集信息
    dataset_info = {
        'num_classes': 10,
        'train_size': len(train_paths),
        'val_size': len(val_paths),
        'ood_size': len(ood_image_paths),
        'ood_categories': ood_categories,
        'train_paths': [str(p) for p in train_paths],
        'val_paths': [str(p) for p in val_paths],
        'train_labels': train_labels_split,
        'val_labels': val_labels_split,
        'ood_paths': [str(p) for p in ood_image_paths],
        'note': '训练标签是基于文件名哈希的伪标签，需要人工标注真实类别'
    }
    
    # 保存到JSON文件
    with open(dataset_dir / "dataset_info.json", 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    print(f"数据集信息已保存到: {dataset_dir / 'dataset_info.json'}")
    
    return dataset_info

def create_dataloaders(dataset_info, batch_size=32, image_size=224):
    """创建数据加载器"""
    
    # 获取变换
    train_transform = get_transforms(image_size, is_training=True)
    val_transform = get_transforms(image_size, is_training=False)
    
    # 创建数据集
    train_dataset = SeedDataset(
        image_paths=[Path(p) for p in dataset_info['train_paths']],
        labels=dataset_info['train_labels'],
        transform=train_transform,
        is_ood=False
    )
    
    val_dataset = SeedDataset(
        image_paths=[Path(p) for p in dataset_info['val_paths']],
        labels=dataset_info['val_labels'],
        transform=val_transform,
        is_ood=False
    )
    
    ood_dataset = SeedDataset(
        image_paths=[Path(p) for p in dataset_info['ood_paths']],
        labels=None,
        transform=val_transform,
        is_ood=True
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    ood_loader = DataLoader(
        ood_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, ood_loader

def test_dataloader(dataloader, name, max_batches=2):
    """测试数据加载器"""
    print(f"\n=== 测试 {name} 数据加载器 ===")
    
    for i, (images, labels) in enumerate(dataloader):
        print(f"批次 {i+1}:")
        print(f"  图像形状: {images.shape}")
        print(f"  标签形状: {labels.shape}")
        print(f"  图像数据类型: {images.dtype}")
        print(f"  图像值范围: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  标签: {labels[:5].tolist()}...")  # 显示前5个标签
        
        if i >= max_batches - 1:
            break
    
    print(f"✅ {name} 数据加载器测试完成")

def main():
    """主函数"""
    print("开始准备种子数据集...")
    
    # 组织数据集
    dataset_info = organize_dataset()
    
    print(f"\n=== 数据集统计 ===")
    print(f"类别数: {dataset_info['num_classes']}")
    print(f"训练集: {dataset_info['train_size']} 张")
    print(f"验证集: {dataset_info['val_size']} 张")
    print(f"OOD测试集: {dataset_info['ood_size']} 张")
    
    # 创建数据加载器
    print(f"\n=== 创建数据加载器 ===")
    train_loader, val_loader, ood_loader = create_dataloaders(dataset_info, batch_size=8)
    
    # 测试数据加载器
    test_dataloader(train_loader, "训练集", max_batches=1)
    test_dataloader(val_loader, "验证集", max_batches=1)
    test_dataloader(ood_loader, "OOD测试集", max_batches=1)
    
    print(f"\n✅ 数据集准备完成！")
    print(f"\n⚠️  重要提醒:")
    print(f"  - 当前使用的是基于文件名的伪标签")
    print(f"  - 为了获得最佳性能，建议人工标注真实的种子类别")
    print(f"  - 数据集信息已保存，可以直接用于模型训练")
    
    return train_loader, val_loader, ood_loader

if __name__ == "__main__":
    main() 