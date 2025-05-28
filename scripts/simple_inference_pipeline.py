#!/usr/bin/env python3
"""
简化版种子分类和OOD检测推理管道
移除复杂依赖，专注于核心推理功能
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass, asdict
import time
import argparse

# OpenOOD imports
from openood.networks import ResNet18_224x224

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """检测结果数据类"""
    filename: str
    predicted_class: str
    predicted_class_name: str
    confidence: float
    is_ood: bool
    probabilities: Dict[str, float]
    processing_time: float
    method: str = "ODIN"
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return asdict(self)

class SeedOODClassifier(torch.nn.Module):
    """种子OOD分类器"""
    
    def __init__(self, num_classes=10):
        super().__init__()
        self.backbone = ResNet18_224x224(num_classes=num_classes)
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)

class SimpleInferencePipeline:
    """简化版推理管道"""
    
    def __init__(self, 
                 model_path: str = "models/best_seed_ood_classifier.pth",
                 eval_results_path: str = "models/ood_evaluation_results.json",
                 device: str = "auto",
                 odin_temperature: float = 1000.0,
                 odin_epsilon: float = 0.0014,
                 ood_threshold: float = 0.2591):
        
        self.model_path = model_path
        self.eval_results_path = eval_results_path
        self.odin_temperature = odin_temperature
        self.odin_epsilon = odin_epsilon
        self.ood_threshold = ood_threshold
        
        # 设备设置
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        logger.info(f"使用设备: {self.device}")
        
        # 加载模型配置
        self._load_eval_results()
        
        # 加载模型
        self._load_model()
        
        # 设置数据变换
        self._setup_transforms()
        
        logger.info("推理管道初始化完成")
    
    def _load_eval_results(self):
        """加载评估结果"""
        try:
            with open(self.eval_results_path, 'r', encoding='utf-8') as f:
                eval_results = json.load(f)
            
            self.num_classes = eval_results['num_classes']
            self.class_to_idx = eval_results['class_to_idx']
            self.idx_to_class = {v: k for k, v in self.class_to_idx.items()}
            
            logger.info(f"加载模型配置: {self.num_classes} 个类别")
            
        except Exception as e:
            logger.error(f"加载评估结果失败: {e}")
            raise
    
    def _load_model(self):
        """加载训练好的模型"""
        try:
            self.model = SeedOODClassifier(num_classes=self.num_classes)
            
            # 加载checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"模型加载完成: {self.model_path}")
            
        except Exception as e:
            logger.error(f"模型加载失败: {e}")
            raise
    
    def _setup_transforms(self):
        """设置数据变换"""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _odin_score(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算ODIN分数"""
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
        """预处理图像"""
        # 加载图像
        image = Image.open(image_path).convert('RGB')
        
        # 应用变换
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor.to(self.device)
    
    def detect_single_image(self, image_path: str) -> DetectionResult:
        """检测单张图像"""
        start_time = time.time()
        filename = Path(image_path).name
        
        try:
            # 预处理图像
            image_tensor = self.preprocess_image(image_path)
            
            # 计算ODIN分数
            logits, odin_scores = self._odin_score(image_tensor)
            
            # 获取预测结果
            with torch.no_grad():
                probabilities = F.softmax(logits, dim=1)
                predicted_idx = probabilities.argmax(dim=1).item()
                predicted_class = str(predicted_idx)
                predicted_class_name = self.idx_to_class.get(predicted_idx, f"Unknown_{predicted_idx}")
                confidence = probabilities.max().item()
                
                # OOD检测
                odin_score = odin_scores.item()
                is_ood = odin_score < self.ood_threshold
            
            processing_time = time.time() - start_time
            
            # 构建概率字典
            prob_dict = {
                self.idx_to_class.get(i, f"class_{i}"): float(probabilities[0][i])
                for i in range(self.num_classes)
            }
            
            return DetectionResult(
                filename=filename,
                predicted_class=predicted_class,
                predicted_class_name=predicted_class_name,
                confidence=confidence,
                is_ood=is_ood,
                probabilities=prob_dict,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            logger.error(f"处理图像失败: {image_path}, 错误: {e}")
            
            return DetectionResult(
                filename=filename,
                predicted_class="ERROR",
                predicted_class_name="ERROR",
                confidence=0.0,
                is_ood=True,
                probabilities={},
                processing_time=processing_time,
                error=str(e)
            )
    
    def detect_batch(self, image_paths: List[str]) -> List[DetectionResult]:
        """批量检测图像"""
        logger.info(f"开始批量检测 {len(image_paths)} 张图像")
        
        results = []
        for i, image_path in enumerate(image_paths):
            result = self.detect_single_image(image_path)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"已处理 {i + 1}/{len(image_paths)} 张图像")
        
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        id_samples = sum(1 for r in successful_results if not r.is_ood)
        ood_samples = sum(1 for r in successful_results if r.is_ood)
        
        logger.info(f"批量检测完成，共处理 {len(results)} 张图像")
        logger.info(f"成功: {len(successful_results)}, 失败: {len(failed_results)}")
        logger.info(f"ID样本: {id_samples}, OOD样本: {ood_samples}")
        
        return results

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="简化版种子分类和OOD检测推理管道")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--output", type=str, default="models/simple_inference_results.json", help="输出结果文件路径")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], default="auto", help="指定设备")
    parser.add_argument("--threshold", type=float, default=0.2591, help="OOD检测阈值")
    
    args = parser.parse_args()
    
    # 初始化推理管道
    pipeline = SimpleInferencePipeline(
        device=args.device,
        ood_threshold=args.threshold
    )
    
    # 准备输入图像列表
    input_path = Path(args.input)
    image_paths = []
    
    if input_path.is_file():
        image_paths = [str(input_path)]
    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        for ext in image_extensions:
            image_paths.extend(input_path.glob(f"**/*{ext}"))
            image_paths.extend(input_path.glob(f"**/*{ext.upper()}"))
        image_paths = [str(p) for p in image_paths]
    else:
        logger.error(f"输入路径不存在: {args.input}")
        return
    
    if not image_paths:
        logger.error("未找到图像文件")
        return
    
    logger.info(f"找到 {len(image_paths)} 张图像")
    
    # 批量检测
    results = pipeline.detect_batch(image_paths)
    
    # 保存结果
    output_data = {
        'total_images': len(image_paths),
        'successful_images': len([r for r in results if r.error is None]),
        'failed_images': len([r for r in results if r.error is not None]),
        'results': [result.to_dict() for result in results]
    }
    
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, ensure_ascii=False, indent=2)
    
    logger.info(f"检测结果已保存到: {args.output}")
    
    # 打印摘要
    successful_results = [r for r in results if r.error is None]
    id_samples = sum(1 for r in successful_results if not r.is_ood)
    ood_samples = sum(1 for r in successful_results if r.is_ood)
    
    print(f"\n🔍 检测结果摘要:")
    print(f"  总图像数: {len(image_paths)}")
    print(f"  成功处理: {len(successful_results)}")
    print(f"  ID样本: {id_samples}")
    print(f"  OOD样本: {ood_samples}")

if __name__ == "__main__":
    main() 