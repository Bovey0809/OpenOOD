#!/usr/bin/env python3
"""
生产就绪的种子OOD检测器
使用优化后的ODIN方法，提供完整的种子分类和异常检测功能
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
import time

# OpenOOD imports
from openood.networks import ResNet18_224x224

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据类"""
    filename: str
    predicted_class: str
    confidence: float
    is_ood: bool
    probabilities: Dict[str, float]
    processing_time: float
    method: str = "ODIN"

class SeedOODClassifier(torch.nn.Module):
    """种子OOD分类器"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ResNet18_224x224(num_classes=num_classes)
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)
    
    def get_features(self, x):
        """获取特征表示"""
        # 获取倒数第二层的特征
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

class ProductionOODDetector:
    """生产环境OOD检测器"""
    
    def __init__(self, 
                 model_path: str = "models/best_seed_ood_classifier.pth",
                 eval_results_path: str = "models/ood_evaluation_results.json",
                 device: str = "auto"):
        """
        初始化OOD检测器
        
        Args:
            model_path: 模型文件路径
            eval_results_path: 评估结果文件路径
            device: 设备类型 ("auto", "cuda", "cpu")
        """
        
        # 设备设置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载评估结果
        self._load_eval_results(eval_results_path)
        
        # 加载模型
        self._load_model(model_path)
        
        # 设置数据变换
        self._setup_transforms()
        
        # ODIN参数 (基于改进实验的最优参数)
        self.odin_temperature = 1000.0
        self.odin_epsilon = 0.0014
        self.ood_threshold = 0.2591  # 基于改进实验的最优阈值
        
        logger.info("生产环境OOD检测器初始化完成")
    
    def _load_eval_results(self, eval_results_path: str):
        """加载评估结果"""
        with open(eval_results_path, 'r', encoding='utf-8') as f:
            eval_results = json.load(f)
        
        self.num_classes = eval_results['num_classes']
        self.class_to_idx = eval_results['class_to_idx']
        self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
        
        logger.info(f"加载模型配置: {self.num_classes} 个类别")
    
    def _load_model(self, model_path: str):
        """加载训练好的模型"""
        self.model = SeedOODClassifier(num_classes=self.num_classes)
        
        # 加载checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # 处理不同的checkpoint格式
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"模型加载完成: {model_path}")
    
    def _setup_transforms(self):
        """设置数据变换"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _odin_score(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算ODIN分数
        
        Args:
            inputs: 输入图像张量
            
        Returns:
            logits: 模型输出
            odin_scores: ODIN分数
        """
        inputs.requires_grad_(True)
        
        # 前向传播
        logits = self.model(inputs)
        
        # 计算最大logit对应的损失
        max_logits = logits.max(dim=1)[0]
        loss = max_logits.sum()
        
        # 反向传播获取梯度
        loss.backward()
        
        # 添加对抗性扰动
        gradient = inputs.grad.data
        # 计算每个样本的梯度范数
        gradient_norms = torch.sqrt(torch.sum(gradient ** 2, dim=(1, 2, 3), keepdim=True))
        gradient = gradient / (gradient_norms + 1e-8)
        
        # 生成对抗样本
        perturbed_inputs = inputs - self.odin_epsilon * gradient
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        # 在扰动输入上计算温度缩放的logits
        with torch.no_grad():
            perturbed_logits = self.model(perturbed_inputs)
            scaled_logits = perturbed_logits / self.odin_temperature
            odin_scores = F.softmax(scaled_logits, dim=1).max(dim=1)[0]
        
        return logits, odin_scores
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        预处理图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理后的图像张量
        """
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def detect_single_image(self, image_path: str) -> DetectionResult:
        """
        检测单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            检测结果
        """
        start_time = time.time()
        
        # 预处理图像
        image_tensor = self.preprocess_image(image_path)
        
        # 计算ODIN分数 (需要在no_grad外部计算)
        logits, odin_scores = self._odin_score(image_tensor)
        
        # 获取预测结果
        with torch.no_grad():
            probabilities = F.softmax(logits, dim=1)
            predicted_idx = probabilities.argmax(dim=1).item()
            predicted_class = self.idx_to_class[predicted_idx]
            confidence = probabilities.max().item()
            
            # OOD检测
            odin_score = odin_scores.item()
            is_ood = odin_score < self.ood_threshold
        
        processing_time = time.time() - start_time
        
        # 构建概率字典
        prob_dict = {
            self.idx_to_class[i]: float(probabilities[0][i])
            for i in range(self.num_classes)
        }
        
        return DetectionResult(
            filename=Path(image_path).name,
            predicted_class=predicted_class,
            confidence=confidence,
            is_ood=is_ood,
            probabilities=prob_dict,
            processing_time=processing_time
        )
    
    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """
        批量检测图像
        
        Args:
            image_paths: 图像文件路径列表
            
        Returns:
            检测结果列表
        """
        results = []
        
        logger.info(f"开始批量检测 {len(image_paths)} 张图像")
        
        for i, image_path in enumerate(image_paths):
            try:
                result = self.detect_single_image(image_path)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(image_paths)} 张图像")
                    
            except Exception as e:
                logger.error(f"处理图像 {image_path} 时出错: {e}")
                # 创建错误结果
                error_result = DetectionResult(
                    filename=Path(image_path).name,
                    predicted_class="ERROR",
                    confidence=0.0,
                    is_ood=True,
                    probabilities={},
                    processing_time=0.0
                )
                results.append(error_result)
        
        logger.info(f"批量检测完成，共处理 {len(results)} 张图像")
        return results
    
    def save_results(self, results: List[DetectionResult], output_path: str):
        """
        保存检测结果
        
        Args:
            results: 检测结果列表
            output_path: 输出文件路径
        """
        # 转换为可序列化的格式
        serializable_results = []
        for result in results:
            serializable_results.append({
                'filename': result.filename,
                'predicted_class': result.predicted_class,
                'confidence': result.confidence,
                'is_ood': result.is_ood,
                'probabilities': result.probabilities,
                'processing_time': result.processing_time,
                'method': result.method
            })
        
        # 计算统计信息
        total_images = len(results)
        ood_count = sum(1 for r in results if r.is_ood)
        id_count = total_images - ood_count
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        # 构建完整结果
        full_results = {
            'summary': {
                'total_images': total_images,
                'id_samples': id_count,
                'ood_samples': ood_count,
                'ood_rate': ood_count / total_images if total_images > 0 else 0,
                'avg_processing_time': avg_processing_time,
                'method': 'ODIN',
                'threshold': self.ood_threshold
            },
            'detailed_results': serializable_results
        }
        
        # 保存到文件
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(full_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"检测结果已保存到: {output_path}")
    
    def print_summary(self, results: List[DetectionResult]):
        """打印检测结果摘要"""
        total_images = len(results)
        ood_count = sum(1 for r in results if r.is_ood)
        id_count = total_images - ood_count
        avg_processing_time = np.mean([r.processing_time for r in results])
        
        print(f"\n🔍 检测结果摘要:")
        print(f"  总图像数: {total_images}")
        print(f"  ID样本: {id_count} ({id_count/total_images:.1%})")
        print(f"  OOD样本: {ood_count} ({ood_count/total_images:.1%})")
        print(f"  平均处理时间: {avg_processing_time:.3f}秒/图像")
        print(f"  检测方法: ODIN")
        print(f"  OOD阈值: {self.ood_threshold:.4f}")

def main():
    """主函数 - 演示用法"""
    
    # 初始化检测器
    detector = ProductionOODDetector()
    
    # 测试图像路径
    test_images = []
    
    # 添加ID样本 (从分割的种子中选择一些)
    segmented_dir = Path("datasets/seeds/segmented")
    if segmented_dir.exists():
        seed_images = list(segmented_dir.glob("*.jpg"))[:10]  # 取前10张
        test_images.extend([str(p) for p in seed_images])
    
    # 添加OOD样本
    ood_dir = Path("竞品种子")
    if ood_dir.exists():
        for category_dir in ood_dir.iterdir():
            if category_dir.is_dir():
                category_images = list(category_dir.glob("*.jpg"))[:2]  # 每类取2张
                test_images.extend([str(p) for p in category_images])
    
    if not test_images:
        logger.warning("未找到测试图像，请检查数据路径")
        return
    
    logger.info(f"找到 {len(test_images)} 张测试图像")
    
    # 批量检测
    results = detector.detect_batch(test_images)
    
    # 打印摘要
    detector.print_summary(results)
    
    # 保存结果
    detector.save_results(results, "models/production_ood_results.json")
    
    # 打印一些详细结果
    print(f"\n📋 详细结果示例:")
    for i, result in enumerate(results[:5]):  # 显示前5个结果
        status = "🚨 OOD" if result.is_ood else "✅ ID"
        print(f"  {i+1}. {result.filename}: {status} | "
              f"类别: {result.predicted_class} | "
              f"置信度: {result.confidence:.3f} | "
              f"时间: {result.processing_time:.3f}s")

if __name__ == "__main__":
    main() 