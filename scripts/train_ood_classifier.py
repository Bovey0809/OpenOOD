#!/usr/bin/env python3
"""
种子OOD检测分类器训练脚本
使用分割后的种子数据和真实类别标签训练分类模型
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict

# OpenOOD imports
from openood.networks import ResNet18_224x224

class SegmentedSeedDataset(Dataset):
    """分割种子数据集类"""
    
    def __init__(self, segmentation_info_path, transform=None, target_size=(224, 224)):
        """
        初始化数据集
        
        Args:
            segmentation_info_path: 分割信息JSON文件路径
            transform: 数据变换
            target_size: 目标图像尺寸
        """
        self.target_size = target_size
        self.transform = transform
        
        # 加载分割信息
        with open(segmentation_info_path, 'r', encoding='utf-8') as f:
            self.segmentation_info = json.load(f)
        
        self.seeds_info = self.segmentation_info['seeds_info']
        
        # 获取类别信息
        self.class_distribution = self.segmentation_info['class_distribution']
        self.unique_classes = sorted(self.segmentation_info['unique_classes'])
        self.num_classes = len(self.unique_classes)
        
        # 创建类别到索引的映射
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
        self.idx_to_class = {idx: cls for cls, idx in self.class_to_idx.items()}
        
        print(f"📊 数据集信息:")
        print(f"  总种子数: {len(self.seeds_info)}")
        print(f"  类别数: {self.num_classes}")
        print(f"  类别分布: {self.class_distribution}")
        print(f"  类别映射: {self.class_to_idx}")
        
    def __len__(self):
        return len(self.seeds_info)
    
    def __getitem__(self, idx):
        seed_info = self.seeds_info[idx]
        
        # 加载种子图像
        image_path = seed_info['path']
        image = Image.open(image_path).convert('RGB')
        
        # 调整图像大小
        image = image.resize(self.target_size, Image.Resampling.LANCZOS)
        
        # 应用变换
        if self.transform:
            image = self.transform(image)
        
        # 获取真实类别标签
        seed_class = seed_info['seed_class']
        label = self.class_to_idx[seed_class]
        
        return image, label, seed_info['filename']

def create_data_transforms():
    """创建数据变换"""
    
    # 训练时的数据增强
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 验证/测试时的变换
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def create_data_loaders(segmentation_info_path, batch_size=32, val_split=0.2):
    """创建数据加载器"""
    
    train_transform, val_transform = create_data_transforms()
    
    # 创建完整数据集
    full_dataset = SegmentedSeedDataset(segmentation_info_path, transform=None)
    
    # 分割数据集
    total_size = len(full_dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    
    train_indices, val_indices = random_split(
        range(total_size), 
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # 创建训练和验证数据集
    train_dataset = SegmentedSeedDataset(segmentation_info_path, transform=train_transform)
    val_dataset = SegmentedSeedDataset(segmentation_info_path, transform=val_transform)
    
    # 使用索引创建子集
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_subset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True
    )
    
    print(f"📊 数据分割:")
    print(f"  训练集: {len(train_subset)} 个种子")
    print(f"  验证集: {len(val_subset)} 个种子")
    
    return train_loader, val_loader, full_dataset.num_classes, full_dataset.class_to_idx

class SeedOODClassifier(nn.Module):
    """种子OOD分类器"""
    
    def __init__(self, num_classes, pretrained=False):
        super(SeedOODClassifier, self).__init__()
        
        # 使用OpenOOD的ResNet18
        self.backbone = ResNet18_224x224(num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

class SeedOODTrainer:
    """种子OOD分类器训练器"""
    
    def __init__(self, model, train_loader, val_loader, num_classes, class_to_idx, device):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_classes = num_classes
        self.class_to_idx = class_to_idx
        self.idx_to_class = {v: k for k, v in class_to_idx.items()}
        self.device = device
        
        # 损失函数和优化器
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # 训练历史
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
    def train_epoch(self):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, (images, labels, _) in enumerate(pbar):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # 前向传播
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # 反向传播
            loss.backward()
            self.optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix({
                'Loss': f'{running_loss/(batch_idx+1):.4f}',
                'Acc': f'{100.*correct/total:.2f}%'
            })
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate_epoch(self):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels, _ in tqdm(self.val_loader, desc="Validation"):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, num_epochs=30):
        """训练模型"""
        print(f"🚀 开始训练，共 {num_epochs} 个epoch")
        print(f"📱 使用设备: {self.device}")
        print(f"🎯 类别数: {self.num_classes}")
        
        best_val_acc = 0.0
        best_model_path = "models/best_seed_ood_classifier.pth"
        
        # 确保模型目录存在
        Path("models").mkdir(exist_ok=True)
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch+1}/{num_epochs}")
            
            # 训练
            train_loss, train_acc = self.train_epoch()
            
            # 验证
            val_loss, val_acc = self.validate_epoch()
            
            # 更新学习率
            self.scheduler.step()
            
            # 记录历史
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            # 打印结果
            print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
            print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
            print(f"学习率: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # 保存最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'num_classes': self.num_classes,
                    'class_to_idx': self.class_to_idx
                }, best_model_path)
                print(f"💾 保存最佳模型 (验证准确率: {val_acc:.2f}%)")
        
        print(f"\n🎉 训练完成！最佳验证准确率: {best_val_acc:.2f}%")
        return best_model_path
    
    def plot_training_history(self, save_path="models/training_history.png"):
        """绘制训练历史"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 损失曲线
        ax1.plot(self.train_losses, label='训练损失', color='blue')
        ax1.plot(self.val_losses, label='验证损失', color='red')
        ax1.set_title('训练和验证损失')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(self.train_accuracies, label='训练准确率', color='blue')
        ax2.plot(self.val_accuracies, label='验证准确率', color='red')
        ax2.set_title('训练和验证准确率')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"📊 训练历史图表已保存到: {save_path}")
        
        try:
            plt.show()
        except:
            print("⚠️  无法显示图像，但已保存到文件")

def evaluate_ood_detection(model, val_loader, device, class_to_idx):
    """评估OOD检测性能"""
    model.eval()
    
    all_predictions = []
    all_labels = []
    all_confidences = []
    all_filenames = []
    
    with torch.no_grad():
        for images, labels, filenames in tqdm(val_loader, desc="OOD评估"):
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)
            confidences = torch.max(probabilities, dim=1)[0]
            predictions = torch.argmax(outputs, dim=1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())
            all_filenames.extend(filenames)
    
    # 计算分类报告
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [f"类别{idx_to_class[i]}" for i in range(len(class_to_idx))]
    
    print("\n📊 分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig("models/confusion_matrix.png", dpi=150, bbox_inches='tight')
    print("📊 混淆矩阵已保存到: models/confusion_matrix.png")
    
    # 置信度分析
    confidence_threshold = np.percentile(all_confidences, 5)  # 使用5%分位数作为阈值
    
    print(f"\n🎯 OOD检测分析:")
    print(f"平均置信度: {np.mean(all_confidences):.4f}")
    print(f"置信度标准差: {np.std(all_confidences):.4f}")
    print(f"建议OOD阈值: {confidence_threshold:.4f}")
    
    # 按类别分析置信度
    class_confidences = defaultdict(list)
    for pred, conf in zip(all_predictions, all_confidences):
        class_confidences[pred].append(conf)
    
    print(f"\n📈 各类别置信度分析:")
    for class_idx in sorted(class_confidences.keys()):
        class_conf = class_confidences[class_idx]
        print(f"  类别{idx_to_class[class_idx]}: 平均={np.mean(class_conf):.4f}, "
              f"标准差={np.std(class_conf):.4f}, 样本数={len(class_conf)}")
    
    return {
        'predictions': all_predictions,
        'labels': all_labels,
        'confidences': all_confidences,
        'filenames': all_filenames,
        'ood_threshold': confidence_threshold
    }

def main():
    """主函数"""
    print("🌱 种子OOD检测分类器训练")
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"🔥 GPU信息: {torch.cuda.get_device_name()}")
        print(f"💾 GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # 数据路径
    segmentation_info_path = "datasets/seeds/segmented/segmentation_info.json"
    
    if not Path(segmentation_info_path).exists():
        print(f"❌ 分割信息文件不存在: {segmentation_info_path}")
        print("请先运行 python scripts/segment_seeds.py")
        return
    
    # 创建数据加载器
    print("\n📊 准备数据...")
    train_loader, val_loader, num_classes, class_to_idx = create_data_loaders(
        segmentation_info_path, 
        batch_size=32, 
        val_split=0.2
    )
    
    # 创建模型
    print(f"\n🏗️  创建模型 (类别数: {num_classes})...")
    model = SeedOODClassifier(num_classes=num_classes)
    model = model.to(device)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"📊 模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")
    
    # 创建训练器
    trainer = SeedOODTrainer(model, train_loader, val_loader, num_classes, class_to_idx, device)
    
    # 训练模型
    best_model_path = trainer.train(num_epochs=30)
    
    # 绘制训练历史
    trainer.plot_training_history()
    
    # 加载最佳模型进行评估
    print("\n📊 加载最佳模型进行评估...")
    checkpoint = torch.load(best_model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 评估OOD检测性能
    ood_results = evaluate_ood_detection(model, val_loader, device, class_to_idx)
    
    # 保存评估结果
    results_path = "models/ood_evaluation_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        # 转换numpy数组为列表以便JSON序列化
        json_results = {
            'ood_threshold': float(ood_results['ood_threshold']),
            'num_classes': num_classes,
            'class_to_idx': class_to_idx,
            'predictions': [int(x) for x in ood_results['predictions']],
            'labels': [int(x) for x in ood_results['labels']],
            'confidences': [float(x) for x in ood_results['confidences']],
            'filenames': ood_results['filenames']
        }
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 评估结果已保存到: {results_path}")
    print(f"🎉 训练和评估完成！")
    print(f"📁 模型文件: {best_model_path}")

if __name__ == "__main__":
    main() 