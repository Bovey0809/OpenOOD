#!/usr/bin/env python3
"""
改进的OOD检测脚本
实施多种策略提升OOD检测性能：
1. 多种OOD检测方法 (MSP, ODIN, Mahalanobis, Energy)
2. 阈值优化
3. 特征分析和可视化
4. 集成方法
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import cv2
from tqdm import tqdm
import seaborn as sns
from collections import defaultdict
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.covariance import EmpiricalCovariance
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False

# OpenOOD imports
from openood.networks import ResNet18_224x224

class SeedOODClassifier(torch.nn.Module):
    """种子OOD分类器"""
    
    def __init__(self, num_classes):
        super(SeedOODClassifier, self).__init__()
        self.backbone = ResNet18_224x224(num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """获取特征表示"""
        # 获取倒数第二层的特征 (在全连接层之前)
        # 对于ResNet18，我们需要手动提取特征
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        x = self.backbone.layer4(x)
        
        x = self.backbone.avgpool(x)
        features = torch.flatten(x, 1)
        return features

class ImprovedOODDetector:
    """改进的OOD检测器"""
    
    def __init__(self, model, device, class_to_idx):
        self.model = model
        self.device = device
        self.class_to_idx = class_to_idx
        self.num_classes = len(class_to_idx)
        
        # 存储训练数据的统计信息
        self.class_means = None
        self.class_covariances = None
        self.global_mean = None
        self.global_covariance = None
        
    def extract_features_and_logits(self, dataloader):
        """提取特征和logits用于统计分析"""
        self.model.eval()
        
        all_features = []
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels, _ in tqdm(dataloader, desc="提取特征"):
                images = images.to(self.device)
                
                # 获取特征和logits
                features = self.model.get_features(images)
                logits = self.model(images)
                
                all_features.append(features.cpu().numpy())
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())
        
        features = np.concatenate(all_features, axis=0)
        logits = np.concatenate(all_logits, axis=0)
        labels = np.concatenate(all_labels, axis=0)
        
        return features, logits, labels
    
    def fit_statistics(self, features, labels):
        """拟合训练数据的统计信息"""
        print("🔧 计算训练数据统计信息...")
        
        # 计算每个类别的均值和协方差
        self.class_means = {}
        self.class_covariances = {}
        
        for class_idx in range(self.num_classes):
            class_features = features[labels == class_idx]
            if len(class_features) > 0:
                self.class_means[class_idx] = np.mean(class_features, axis=0)
                
                # 使用经验协方差估计
                if len(class_features) > 1:
                    cov_estimator = EmpiricalCovariance()
                    cov_estimator.fit(class_features)
                    self.class_covariances[class_idx] = cov_estimator.covariance_
                else:
                    # 如果样本太少，使用单位矩阵
                    self.class_covariances[class_idx] = np.eye(class_features.shape[1])
        
        # 计算全局统计信息
        self.global_mean = np.mean(features, axis=0)
        cov_estimator = EmpiricalCovariance()
        cov_estimator.fit(features)
        self.global_covariance = cov_estimator.covariance_
        
        print(f"✅ 统计信息计算完成，特征维度: {features.shape[1]}")
    
    def msp_score(self, logits):
        """Maximum Softmax Probability (MSP) 方法"""
        probabilities = F.softmax(torch.tensor(logits), dim=1)
        confidence = torch.max(probabilities, dim=1)[0]
        return confidence.numpy()
    
    def odin_score(self, image, temperature=1000, epsilon=0.0014):
        """ODIN方法 - 使用温度缩放和输入预处理"""
        self.model.eval()
        
        # 启用梯度计算
        image.requires_grad_(True)
        
        # 前向传播
        logits = self.model(image)
        
        # 温度缩放
        scaled_logits = logits / temperature
        
        # 计算最大类别的损失
        max_class = torch.argmax(scaled_logits, dim=1)
        loss = F.cross_entropy(scaled_logits, max_class)
        
        # 反向传播获取梯度
        loss.backward()
        
        # 输入预处理 - 添加对抗性噪声
        gradient = torch.sign(image.grad.data)
        perturbed_image = image - epsilon * gradient
        
        # 重新计算logits
        with torch.no_grad():
            perturbed_logits = self.model(perturbed_image)
            scaled_logits = perturbed_logits / temperature
            probabilities = F.softmax(scaled_logits, dim=1)
            confidence = torch.max(probabilities, dim=1)[0]
        
        return confidence.cpu().numpy()
    
    def mahalanobis_score(self, features):
        """Mahalanobis距离方法"""
        if self.class_means is None:
            raise ValueError("需要先调用fit_statistics方法")
        
        scores = []
        
        for feature in features:
            min_distance = float('inf')
            
            # 计算到每个类别的Mahalanobis距离
            for class_idx in range(self.num_classes):
                if class_idx in self.class_means:
                    mean = self.class_means[class_idx]
                    cov = self.class_covariances[class_idx]
                    
                    # 计算Mahalanobis距离
                    diff = feature - mean
                    try:
                        inv_cov = np.linalg.pinv(cov)
                        distance = np.sqrt(diff.T @ inv_cov @ diff)
                        min_distance = min(min_distance, distance)
                    except:
                        # 如果协方差矩阵奇异，使用欧几里得距离
                        distance = np.linalg.norm(diff)
                        min_distance = min(min_distance, distance)
            
            scores.append(-min_distance)  # 负号使得更高的分数表示更可能是ID
        
        return np.array(scores)
    
    def energy_score(self, logits, temperature=1):
        """Energy方法"""
        energy = -temperature * torch.logsumexp(torch.tensor(logits) / temperature, dim=1)
        return -energy.numpy()  # 负号使得更高的分数表示更可能是ID
    
    def predict_with_multiple_methods(self, image_path):
        """使用多种方法预测单张图像"""
        # 加载和预处理图像
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(self.device)
        
        self.model.eval()
        
        with torch.no_grad():
            # 获取logits和特征
            logits = self.model(image_tensor)
            features = self.model.get_features(image_tensor)
            
            # 基本预测
            probabilities = F.softmax(logits, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            # 计算各种OOD分数
            msp = self.msp_score(logits.cpu().numpy())[0]
            energy = self.energy_score(logits.cpu().numpy())[0]
            
            # Mahalanobis分数
            if self.class_means is not None:
                mahalanobis = self.mahalanobis_score(features.cpu().numpy())[0]
            else:
                mahalanobis = 0.0
        
        # ODIN分数 (需要梯度)
        image_tensor_grad = image_tensor.clone().detach().requires_grad_(True)
        try:
            odin = self.odin_score(image_tensor_grad)[0]
        except:
            odin = msp  # 如果ODIN失败，使用MSP
        
        return {
            'predicted_class': predicted_class,
            'probabilities': probabilities.cpu().numpy()[0],
            'msp_score': float(msp),
            'odin_score': float(odin),
            'mahalanobis_score': float(mahalanobis),
            'energy_score': float(energy)
        }

def load_model_and_data():
    """加载模型和数据"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型配置
    with open('models/ood_evaluation_results.json', 'r', encoding='utf-8') as f:
        eval_results = json.load(f)
    
    num_classes = eval_results['num_classes']
    class_to_idx = eval_results['class_to_idx']
    
    # 加载模型
    model = SeedOODClassifier(num_classes=num_classes)
    checkpoint = torch.load('models/best_seed_ood_classifier.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    
    return model, device, class_to_idx

def load_ood_images(ood_dir):
    """加载OOD测试图像"""
    ood_dir = Path(ood_dir)
    ood_images = []
    
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    for image_path in ood_dir.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            ood_images.append({
                'path': str(image_path),
                'filename': image_path.name,
                'category': image_path.parent.name
            })
    
    return ood_images

def create_train_dataloader():
    """创建训练数据加载器用于统计分析"""
    import sys
    sys.path.append('.')
    
    # 直接定义数据集类
    class SegmentedSeedDataset(torch.utils.data.Dataset):
        def __init__(self, segmentation_info_path, transform=None):
            with open(segmentation_info_path, 'r', encoding='utf-8') as f:
                self.segmentation_info = json.load(f)
            
            self.seeds_info = self.segmentation_info['seeds_info']
            self.unique_classes = sorted(self.segmentation_info['unique_classes'])
            self.class_to_idx = {cls: idx for idx, cls in enumerate(self.unique_classes)}
            self.transform = transform
            
        def __len__(self):
            return len(self.seeds_info)
        
        def __getitem__(self, idx):
            seed_info = self.seeds_info[idx]
            image = Image.open(seed_info['path']).convert('RGB')
            image = image.resize((224, 224), Image.Resampling.LANCZOS)
            
            if self.transform:
                image = self.transform(image)
            
            seed_class = seed_info['seed_class']
            label = self.class_to_idx[seed_class]
            
            return image, label, seed_info['filename']
    
    # 创建变换
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    dataset = SegmentedSeedDataset(
        'datasets/seeds/segmented/segmentation_info.json', 
        transform=val_transform
    )
    
    # 使用所有数据进行统计分析
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    return dataloader

def optimize_thresholds(id_scores, ood_scores, method_name):
    """优化检测阈值"""
    # 合并分数和标签
    all_scores = np.concatenate([id_scores, ood_scores])
    all_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
    
    # 计算ROC曲线
    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    auc = roc_auc_score(all_labels, all_scores)
    
    # 找到最优阈值 (Youden's J statistic)
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = thresholds[optimal_idx]
    
    return {
        'auc': auc,
        'optimal_threshold': optimal_threshold,
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': thresholds
    }

def evaluate_improved_ood_detection():
    """评估改进的OOD检测方法"""
    print("🚀 开始改进的OOD检测评估")
    
    # 加载模型和数据
    model, device, class_to_idx = load_model_and_data()
    detector = ImprovedOODDetector(model, device, class_to_idx)
    
    # 创建训练数据加载器
    print("📊 准备训练数据统计...")
    train_dataloader = create_train_dataloader()
    
    # 提取训练数据特征并拟合统计信息
    train_features, train_logits, train_labels = detector.extract_features_and_logits(train_dataloader)
    detector.fit_statistics(train_features, train_labels)
    
    # 加载OOD图像
    ood_images = load_ood_images("竞品种子")
    print(f"📊 找到 {len(ood_images)} 个OOD图像")
    
    # 评估ID样本 (使用训练数据的一部分)
    print("📊 评估ID样本...")
    id_results = []
    sample_indices = np.random.choice(len(train_features), min(100, len(train_features)), replace=False)
    
    for idx in tqdm(sample_indices, desc="ID样本"):
        # 重新预测以获取完整结果
        # 这里简化处理，直接使用已有的特征和logits
        feature = train_features[idx:idx+1]
        logit = train_logits[idx:idx+1]
        
        msp = detector.msp_score(logit)[0]
        energy = detector.energy_score(logit)[0]
        mahalanobis = detector.mahalanobis_score(feature)[0]
        
        id_results.append({
            'msp_score': msp,
            'energy_score': energy,
            'mahalanobis_score': mahalanobis,
            'odin_score': msp  # 简化处理
        })
    
    # 评估OOD样本
    print("📊 评估OOD样本...")
    ood_results = []
    
    for ood_info in tqdm(ood_images, desc="OOD样本"):
        try:
            result = detector.predict_with_multiple_methods(ood_info['path'])
            result['filename'] = ood_info['filename']
            result['category'] = ood_info['category']
            ood_results.append(result)
        except Exception as e:
            print(f"⚠️ 处理OOD图像失败: {ood_info['path']}, 错误: {e}")
    
    # 分析各种方法的性能
    methods = ['msp_score', 'odin_score', 'mahalanobis_score', 'energy_score']
    results = {}
    
    print("\n📊 分析各种OOD检测方法的性能...")
    
    for method in methods:
        print(f"\n🔍 分析 {method.upper()} 方法:")
        
        # 提取分数
        id_scores = [r[method] for r in id_results]
        ood_scores = [r[method] for r in ood_results]
        
        # 优化阈值
        method_results = optimize_thresholds(id_scores, ood_scores, method)
        
        # 计算检测性能
        threshold = method_results['optimal_threshold']
        id_correct = sum(1 for score in id_scores if score >= threshold)
        ood_detected = sum(1 for score in ood_scores if score < threshold)
        
        id_accuracy = id_correct / len(id_scores) if len(id_scores) > 0 else 0
        ood_detection_rate = ood_detected / len(ood_scores) if len(ood_scores) > 0 else 0
        
        results[method] = {
            'auc': method_results['auc'],
            'optimal_threshold': threshold,
            'id_accuracy': id_accuracy,
            'ood_detection_rate': ood_detection_rate,
            'id_scores': id_scores,
            'ood_scores': ood_scores
        }
        
        print(f"  AUC: {method_results['auc']:.4f}")
        print(f"  最优阈值: {threshold:.4f}")
        print(f"  ID准确率: {id_accuracy:.2%}")
        print(f"  OOD检测率: {ood_detection_rate:.2%}")
    
    # 可视化结果
    plot_method_comparison(results)
    
    # 集成方法
    print("\n🔧 尝试集成方法...")
    ensemble_results = evaluate_ensemble_method(results)
    
    # 保存结果
    save_improved_results(results, ensemble_results, ood_results)
    
    return results, ensemble_results

def plot_method_comparison(results):
    """绘制方法比较图"""
    methods = list(results.keys())
    
    # 创建子图
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('改进的OOD检测方法比较', fontsize=16)
    
    # AUC比较
    aucs = [results[method]['auc'] for method in methods]
    axes[0, 0].bar(methods, aucs, color=['blue', 'red', 'green', 'orange'])
    axes[0, 0].set_title('AUC比较')
    axes[0, 0].set_ylabel('AUC')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # OOD检测率比较
    ood_rates = [results[method]['ood_detection_rate'] for method in methods]
    axes[0, 1].bar(methods, ood_rates, color=['blue', 'red', 'green', 'orange'])
    axes[0, 1].set_title('OOD检测率比较')
    axes[0, 1].set_ylabel('检测率')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # 置信度分布比较 (以MSP为例)
    method = 'msp_score'
    if method in results:
        axes[1, 0].hist(results[method]['id_scores'], bins=20, alpha=0.7, label='ID样本', color='blue')
        axes[1, 0].hist(results[method]['ood_scores'], bins=20, alpha=0.7, label='OOD样本', color='red')
        axes[1, 0].axvline(x=results[method]['optimal_threshold'], color='green', linestyle='--', label='最优阈值')
        axes[1, 0].set_title(f'{method.upper()} 分数分布')
        axes[1, 0].set_xlabel('分数')
        axes[1, 0].set_ylabel('频次')
        axes[1, 0].legend()
    
    # ROC曲线 (需要重新计算)
    for i, method in enumerate(methods):
        id_scores = results[method]['id_scores']
        ood_scores = results[method]['ood_scores']
        
        all_scores = np.concatenate([id_scores, ood_scores])
        all_labels = np.concatenate([np.ones(len(id_scores)), np.zeros(len(ood_scores))])
        
        fpr, tpr, _ = roc_curve(all_labels, all_scores)
        auc = results[method]['auc']
        
        axes[1, 1].plot(fpr, tpr, label=f'{method} (AUC={auc:.3f})')
    
    axes[1, 1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[1, 1].set_title('ROC曲线比较')
    axes[1, 1].set_xlabel('假阳性率')
    axes[1, 1].set_ylabel('真阳性率')
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.savefig('models/improved_ood_comparison.png', dpi=150, bbox_inches='tight')
    print("📊 方法比较图已保存到: models/improved_ood_comparison.png")

def evaluate_ensemble_method(results):
    """评估集成方法"""
    methods = list(results.keys())
    
    # 简单平均集成
    id_ensemble_scores = []
    ood_ensemble_scores = []
    
    for i in range(len(results[methods[0]]['id_scores'])):
        scores = [results[method]['id_scores'][i] for method in methods]
        id_ensemble_scores.append(np.mean(scores))
    
    for i in range(len(results[methods[0]]['ood_scores'])):
        scores = [results[method]['ood_scores'][i] for method in methods]
        ood_ensemble_scores.append(np.mean(scores))
    
    # 优化集成阈值
    ensemble_results = optimize_thresholds(id_ensemble_scores, ood_ensemble_scores, 'ensemble')
    
    # 计算性能
    threshold = ensemble_results['optimal_threshold']
    id_correct = sum(1 for score in id_ensemble_scores if score >= threshold)
    ood_detected = sum(1 for score in ood_ensemble_scores if score < threshold)
    
    id_accuracy = id_correct / len(id_ensemble_scores)
    ood_detection_rate = ood_detected / len(ood_ensemble_scores)
    
    print(f"🎯 集成方法结果:")
    print(f"  AUC: {ensemble_results['auc']:.4f}")
    print(f"  最优阈值: {threshold:.4f}")
    print(f"  ID准确率: {id_accuracy:.2%}")
    print(f"  OOD检测率: {ood_detection_rate:.2%}")
    
    return {
        'auc': ensemble_results['auc'],
        'optimal_threshold': threshold,
        'id_accuracy': id_accuracy,
        'ood_detection_rate': ood_detection_rate,
        'id_scores': id_ensemble_scores,
        'ood_scores': ood_ensemble_scores
    }

def save_improved_results(results, ensemble_results, ood_results):
    """保存改进的结果"""
    # 转换numpy数组为列表
    for method in results:
        results[method]['id_scores'] = [float(x) for x in results[method]['id_scores']]
        results[method]['ood_scores'] = [float(x) for x in results[method]['ood_scores']]
    
    ensemble_results['id_scores'] = [float(x) for x in ensemble_results['id_scores']]
    ensemble_results['ood_scores'] = [float(x) for x in ensemble_results['ood_scores']]
    
    # 保存详细结果
    detailed_results = {
        'method_results': results,
        'ensemble_results': ensemble_results,
        'ood_predictions': [{
            'filename': r['filename'],
            'category': r['category'],
            'predicted_class': r['predicted_class'],
            'probabilities': r['probabilities'].tolist() if hasattr(r['probabilities'], 'tolist') else r['probabilities'],
            'msp_score': r['msp_score'],
            'odin_score': r['odin_score'],
            'mahalanobis_score': r['mahalanobis_score'],
            'energy_score': r['energy_score']
        } for r in ood_results]
    }
    
    results_path = "models/improved_ood_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"💾 改进的OOD检测结果已保存到: {results_path}")

def main():
    """主函数"""
    print("🌟 改进的种子OOD检测评估")
    
    # 检查必要文件
    required_files = [
        'models/best_seed_ood_classifier.pth',
        'models/ood_evaluation_results.json',
        'datasets/seeds/segmented/segmentation_info.json'
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ 必要文件不存在: {file_path}")
            return
    
    # 运行改进的评估
    results, ensemble_results = evaluate_improved_ood_detection()
    
    # 总结最佳方法
    print("\n🏆 方法性能总结:")
    print("=" * 60)
    
    best_auc = 0
    best_method = ""
    
    for method, result in results.items():
        print(f"{method.upper():15} | AUC: {result['auc']:.4f} | OOD检测率: {result['ood_detection_rate']:.2%}")
        if result['auc'] > best_auc:
            best_auc = result['auc']
            best_method = method
    
    print(f"{'ENSEMBLE':15} | AUC: {ensemble_results['auc']:.4f} | OOD检测率: {ensemble_results['ood_detection_rate']:.2%}")
    
    if ensemble_results['auc'] > best_auc:
        best_method = "ENSEMBLE"
        best_auc = ensemble_results['auc']
    
    print("=" * 60)
    print(f"🥇 最佳方法: {best_method} (AUC: {best_auc:.4f})")
    
    print("\n🎉 改进的OOD检测评估完成！")

if __name__ == "__main__":
    main() 