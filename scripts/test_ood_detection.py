#!/usr/bin/env python3
"""
OOD检测测试脚本
使用训练好的模型测试竞品种子（外来种子）的检测能力
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

# OpenOOD imports
from openood.networks import ResNet18_224x224

class SeedOODClassifier(torch.nn.Module):
    """种子OOD分类器"""
    
    def __init__(self, num_classes):
        super(SeedOODClassifier, self).__init__()
        self.backbone = ResNet18_224x224(num_classes=num_classes)
        
    def forward(self, x):
        return self.backbone(x)

def load_model(model_path, num_classes, device):
    """加载训练好的模型"""
    model = SeedOODClassifier(num_classes=num_classes)
    
    # 加载checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # 如果checkpoint包含model_state_dict，则使用它；否则直接使用checkpoint
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    return model

def create_test_transform():
    """创建测试数据变换"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def load_ood_images(ood_dir):
    """加载OOD测试图像（竞品种子）"""
    ood_dir = Path(ood_dir)
    ood_images = []
    
    # 支持的图像格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    
    print(f"🔍 扫描OOD图像目录: {ood_dir}")
    
    # 递归搜索所有图像文件
    for image_path in ood_dir.rglob('*'):
        if image_path.suffix.lower() in image_extensions:
            ood_images.append({
                'path': str(image_path),
                'filename': image_path.name,
                'category': image_path.parent.name
            })
    
    print(f"📊 找到 {len(ood_images)} 个OOD图像")
    
    # 按类别统计
    category_counts = defaultdict(int)
    for img in ood_images:
        category_counts[img['category']] += 1
    
    print("📈 OOD图像分布:")
    for category, count in category_counts.items():
        print(f"  {category}: {count} 张")
    
    return ood_images

def predict_with_confidence(model, image, transform, device):
    """预测图像并返回置信度"""
    # 预处理图像
    if isinstance(image, str):
        image = Image.open(image).convert('RGB')
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        confidence = torch.max(probabilities, dim=1)[0].item()
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    return predicted_class, confidence, probabilities.cpu().numpy()[0]

def evaluate_ood_detection(model, id_images, ood_images, transform, device, class_to_idx, threshold=0.3816):
    """评估OOD检测性能"""
    
    print(f"\n🎯 开始OOD检测评估 (阈值: {threshold:.4f})")
    
    # 加载ID数据的分割信息
    with open('datasets/seeds/segmented/segmentation_info.json', 'r', encoding='utf-8') as f:
        segmentation_info = json.load(f)
    
    id_seeds = segmentation_info['seeds_info']
    
    # 评估ID样本
    print("📊 评估ID样本...")
    id_results = []
    
    for seed_info in tqdm(id_seeds[:100], desc="ID样本"):  # 取前100个样本测试
        try:
            predicted_class, confidence, probs = predict_with_confidence(
                model, seed_info['path'], transform, device
            )
            
            id_results.append({
                'filename': seed_info['filename'],
                'true_class': seed_info['seed_class'],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_ood_detected': confidence < threshold,
                'probabilities': probs
            })
        except Exception as e:
            print(f"⚠️ 处理ID图像失败: {seed_info['path']}, 错误: {e}")
    
    # 评估OOD样本
    print("📊 评估OOD样本...")
    ood_results = []
    
    for ood_info in tqdm(ood_images, desc="OOD样本"):
        try:
            predicted_class, confidence, probs = predict_with_confidence(
                model, ood_info['path'], transform, device
            )
            
            ood_results.append({
                'filename': ood_info['filename'],
                'category': ood_info['category'],
                'predicted_class': predicted_class,
                'confidence': confidence,
                'is_ood_detected': confidence < threshold,
                'probabilities': probs
            })
        except Exception as e:
            print(f"⚠️ 处理OOD图像失败: {ood_info['path']}, 错误: {e}")
    
    return id_results, ood_results

def calculate_metrics(id_results, ood_results):
    """计算OOD检测指标"""
    
    # ID样本中被错误检测为OOD的数量（假阳性）
    id_false_positives = sum(1 for r in id_results if r['is_ood_detected'])
    id_total = len(id_results)
    
    # OOD样本中被正确检测为OOD的数量（真阳性）
    ood_true_positives = sum(1 for r in ood_results if r['is_ood_detected'])
    ood_total = len(ood_results)
    
    # 计算指标
    id_accuracy = (id_total - id_false_positives) / id_total if id_total > 0 else 0
    ood_detection_rate = ood_true_positives / ood_total if ood_total > 0 else 0
    
    # 整体准确率
    total_correct = (id_total - id_false_positives) + ood_true_positives
    total_samples = id_total + ood_total
    overall_accuracy = total_correct / total_samples if total_samples > 0 else 0
    
    return {
        'id_accuracy': id_accuracy,
        'ood_detection_rate': ood_detection_rate,
        'overall_accuracy': overall_accuracy,
        'id_false_positives': id_false_positives,
        'id_total': id_total,
        'ood_true_positives': ood_true_positives,
        'ood_total': ood_total
    }

def plot_confidence_distribution(id_results, ood_results, threshold, save_path="models/confidence_distribution.png"):
    """绘制置信度分布图"""
    
    id_confidences = [r['confidence'] for r in id_results]
    ood_confidences = [r['confidence'] for r in ood_results]
    
    plt.figure(figsize=(12, 6))
    
    # 绘制直方图
    plt.hist(id_confidences, bins=30, alpha=0.7, label=f'ID样本 (n={len(id_confidences)})', color='blue')
    plt.hist(ood_confidences, bins=30, alpha=0.7, label=f'OOD样本 (n={len(ood_confidences)})', color='red')
    
    # 添加阈值线
    plt.axvline(x=threshold, color='green', linestyle='--', linewidth=2, label=f'OOD阈值 ({threshold:.4f})')
    
    plt.xlabel('置信度')
    plt.ylabel('频次')
    plt.title('ID vs OOD 置信度分布')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"📊 置信度分布图已保存到: {save_path}")

def analyze_ood_categories(ood_results):
    """分析不同OOD类别的检测效果"""
    
    category_stats = defaultdict(lambda: {'total': 0, 'detected': 0, 'confidences': []})
    
    for result in ood_results:
        category = result['category']
        category_stats[category]['total'] += 1
        category_stats[category]['confidences'].append(result['confidence'])
        if result['is_ood_detected']:
            category_stats[category]['detected'] += 1
    
    print("\n📈 各OOD类别检测效果:")
    for category, stats in category_stats.items():
        detection_rate = stats['detected'] / stats['total'] if stats['total'] > 0 else 0
        avg_confidence = np.mean(stats['confidences'])
        print(f"  {category}: 检测率={detection_rate:.2%}, 平均置信度={avg_confidence:.4f}, 样本数={stats['total']}")
    
    return category_stats

def main():
    """主函数"""
    
    print("🔍 种子OOD检测测试")
    
    # 设备设置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 加载模型配置
    model_path = "models/best_seed_ood_classifier.pth"
    evaluation_results_path = "models/ood_evaluation_results.json"
    
    if not Path(model_path).exists():
        print(f"❌ 模型文件不存在: {model_path}")
        print("请先运行 train_ood_classifier.py 训练模型")
        return
    
    # 加载评估结果获取配置信息
    with open(evaluation_results_path, 'r', encoding='utf-8') as f:
        eval_results = json.load(f)
    
    num_classes = eval_results['num_classes']
    class_to_idx = eval_results['class_to_idx']
    suggested_threshold = eval_results['ood_threshold']
    
    print(f"📊 模型配置: {num_classes} 个类别")
    print(f"🎯 建议OOD阈值: {suggested_threshold:.4f}")
    
    # 加载模型
    print("🏗️ 加载训练好的模型...")
    model = load_model(model_path, num_classes, device)
    
    # 创建数据变换
    transform = create_test_transform()
    
    # 加载OOD图像（竞品种子）
    ood_dir = "竞品种子"
    if not Path(ood_dir).exists():
        print(f"❌ OOD数据目录不存在: {ood_dir}")
        return
    
    ood_images = load_ood_images(ood_dir)
    if len(ood_images) == 0:
        print("❌ 未找到OOD图像")
        return
    
    # 评估OOD检测
    id_results, ood_results = evaluate_ood_detection(
        model, None, ood_images, transform, device, class_to_idx, suggested_threshold
    )
    
    # 计算指标
    metrics = calculate_metrics(id_results, ood_results)
    
    # 打印结果
    print(f"\n🎯 OOD检测结果:")
    print(f"  ID样本准确率: {metrics['id_accuracy']:.2%} ({metrics['id_total'] - metrics['id_false_positives']}/{metrics['id_total']})")
    print(f"  OOD检测率: {metrics['ood_detection_rate']:.2%} ({metrics['ood_true_positives']}/{metrics['ood_total']})")
    print(f"  整体准确率: {metrics['overall_accuracy']:.2%}")
    
    # 分析OOD类别
    category_stats = analyze_ood_categories(ood_results)
    
    # 绘制置信度分布
    plot_confidence_distribution(id_results, ood_results, suggested_threshold)
    
    # 保存详细结果
    detailed_results = {
        'metrics': metrics,
        'threshold': suggested_threshold,
        'id_results': [{
            'filename': r['filename'],
            'true_class': r['true_class'],
            'predicted_class': r['predicted_class'],
            'confidence': float(r['confidence']),
            'is_ood_detected': r['is_ood_detected'],
            'probabilities': r['probabilities'].tolist() if hasattr(r['probabilities'], 'tolist') else r['probabilities']
        } for r in id_results],
        'ood_results': [{
            'filename': r['filename'],
            'category': r['category'],
            'predicted_class': r['predicted_class'],
            'confidence': float(r['confidence']),
            'is_ood_detected': r['is_ood_detected'],
            'probabilities': r['probabilities'].tolist() if hasattr(r['probabilities'], 'tolist') else r['probabilities']
        } for r in ood_results],
        'category_stats': {k: {
            'total': v['total'],
            'detected': v['detected'],
            'detection_rate': v['detected'] / v['total'] if v['total'] > 0 else 0,
            'avg_confidence': float(np.mean(v['confidences']))
        } for k, v in category_stats.items()}
    }
    
    results_path = "models/ood_test_results.json"
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(detailed_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 详细结果已保存到: {results_path}")
    print("🎉 OOD检测测试完成！")

if __name__ == "__main__":
    main() 