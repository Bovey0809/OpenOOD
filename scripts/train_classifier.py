#!/usr/bin/env python3
"""
种子分类模型训练脚本
使用OpenOOD框架训练10类种子分类模型
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# 添加OpenOOD路径
sys.path.append('.')

from openood.networks import get_network
from scripts.prepare_dataset import SeedDataset, get_transforms, create_dataloaders
from openood.utils import Config

class SeedClassifierTrainer:
    """种子分类器训练类"""
    
    def __init__(self, config_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载数据集信息
        self.load_dataset_info()
        
        # 创建模型
        self.create_model()
        
        # 创建数据加载器
        self.create_dataloaders()
        
        # 设置训练参数
        self.setup_training()
        
        # 创建保存目录
        self.save_dir = Path("models/seed_classifier")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
    def load_dataset_info(self):
        """加载数据集信息"""
        dataset_info_path = Path("datasets/seeds/dataset_info.json")
        if not dataset_info_path.exists():
            raise FileNotFoundError("数据集信息文件不存在，请先运行 prepare_dataset.py")
        
        with open(dataset_info_path, 'r', encoding='utf-8') as f:
            self.dataset_info = json.load(f)
        
        self.num_classes = self.dataset_info['num_classes']
        print(f"数据集信息加载完成，类别数: {self.num_classes}")
        
    def create_model(self):
        """创建模型"""
        # 创建网络配置
        class NetworkConfig:
            def __init__(self, num_classes):
                self.name = 'resnet18_224x224'
                self.num_classes = num_classes
                self.pretrained = False
                self.checkpoint = None
                self.num_gpus = 1
        
        network_config = NetworkConfig(self.num_classes)
        
        # 获取网络
        self.model = get_network(network_config)
        self.model = self.model.to(self.device)
        
        print(f"模型创建完成: ResNet18_224x224")
        print(f"模型参数数量: {sum(p.numel() for p in self.model.parameters()):,}")
        
    def create_dataloaders(self):
        """创建数据加载器"""
        self.train_loader, self.val_loader, self.ood_loader = create_dataloaders(
            self.dataset_info, 
            batch_size=32, 
            image_size=224
        )
        
        print(f"数据加载器创建完成:")
        print(f"  训练集: {len(self.train_loader)} 批次")
        print(f"  验证集: {len(self.val_loader)} 批次")
        print(f"  OOD测试集: {len(self.ood_loader)} 批次")
        
    def setup_training(self):
        """设置训练参数"""
        # 损失函数
        self.criterion = nn.CrossEntropyLoss()
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=0.001, 
            weight_decay=1e-4
        )
        
        # 学习率调度器
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, 
            step_size=10, 
            gamma=0.1
        )
        
        # 训练历史
        self.train_history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        print("训练参数设置完成")
        
    def train_epoch(self, epoch):
        """训练一个epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {epoch+1} [Train]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
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
    
    def validate_epoch(self, epoch):
        """验证一个epoch"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {epoch+1} [Val]')
            
            for batch_idx, (images, labels) in enumerate(pbar):
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                # 更新进度条
                pbar.set_postfix({
                    'Loss': f'{running_loss/(batch_idx+1):.4f}',
                    'Acc': f'{100.*correct/total:.2f}%'
                })
        
        epoch_loss = running_loss / len(self.val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def save_checkpoint(self, epoch, is_best=False):
        """保存检查点"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_history': self.train_history,
            'dataset_info': self.dataset_info
        }
        
        # 保存最新检查点
        torch.save(checkpoint, self.save_dir / 'latest_checkpoint.pth')
        
        # 保存最佳模型
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best_model.pth')
            print(f"✅ 保存最佳模型 (Epoch {epoch+1})")
    
    def plot_training_history(self):
        """绘制训练历史"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
        
        epochs = range(1, len(self.train_history['train_loss']) + 1)
        
        # 损失曲线
        ax1.plot(epochs, self.train_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.train_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Loss Curves')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # 准确率曲线
        ax2.plot(epochs, self.train_history['train_acc'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.train_history['val_acc'], 'r-', label='Val Acc')
        ax2.set_title('Accuracy Curves')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # 学习率曲线
        ax3.plot(epochs, self.train_history['lr'], 'g-', label='Learning Rate')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('LR')
        ax3.legend()
        ax3.grid(True)
        ax3.set_yscale('log')
        
        # 训练总结
        ax4.text(0.1, 0.8, f"最佳验证准确率: {max(self.train_history['val_acc']):.2f}%", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.7, f"最终训练准确率: {self.train_history['train_acc'][-1]:.2f}%", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.6, f"最终验证准确率: {self.train_history['val_acc'][-1]:.2f}%", 
                transform=ax4.transAxes, fontsize=12)
        ax4.text(0.1, 0.5, f"总训练轮数: {len(epochs)}", 
                transform=ax4.transAxes, fontsize=12)
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_history.png', dpi=150, bbox_inches='tight')
        print(f"📊 训练历史图表已保存到: {self.save_dir / 'training_history.png'}")
    
    def train(self, num_epochs=50):
        """训练模型"""
        print(f"\n🚀 开始训练种子分类模型 (共 {num_epochs} 轮)")
        print("="*60)
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(num_epochs):
            # 训练
            train_loss, train_acc = self.train_epoch(epoch)
            
            # 验证
            val_loss, val_acc = self.validate_epoch(epoch)
            
            # 更新学习率
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # 记录历史
            self.train_history['train_loss'].append(train_loss)
            self.train_history['train_acc'].append(train_acc)
            self.train_history['val_loss'].append(val_loss)
            self.train_history['val_acc'].append(val_acc)
            self.train_history['lr'].append(current_lr)
            
            # 打印结果
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:6.2f}% | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:6.2f}% | "
                  f"LR: {current_lr:.6f}")
            
            # 保存最佳模型
            is_best = val_acc > best_val_acc
            if is_best:
                best_val_acc = val_acc
            
            # 每5轮保存一次检查点
            if (epoch + 1) % 5 == 0 or is_best:
                self.save_checkpoint(epoch, is_best)
        
        # 训练完成
        total_time = time.time() - start_time
        print("="*60)
        print(f"✅ 训练完成！")
        print(f"总训练时间: {total_time/3600:.2f} 小时")
        print(f"最佳验证准确率: {best_val_acc:.2f}%")
        
        # 绘制训练历史
        self.plot_training_history()
        
        # 保存最终模型
        self.save_checkpoint(num_epochs-1)
        
        return best_val_acc

def main():
    """主函数"""
    print("🌱 种子分类模型训练")
    print("="*50)
    
    # 创建训练器
    trainer = SeedClassifierTrainer()
    
    # 开始训练
    best_acc = trainer.train(num_epochs=30)
    
    print(f"\n🎉 训练完成！最佳验证准确率: {best_acc:.2f}%")
    print(f"模型已保存到: {trainer.save_dir}")

if __name__ == "__main__":
    main() 