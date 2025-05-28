#!/usr/bin/env python3
"""
完整的种子分类和OOD检测推理管道
提供端到端的推理服务，包含配置管理、性能监控、错误处理等功能
"""

import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import yaml
from pathlib import Path
import cv2
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass, asdict
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import psutil
import GPUtil
from contextlib import contextmanager

# OpenOOD imports
from openood.networks import ResNet18_224x224

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class InferenceConfig:
    """推理配置"""
    model_path: str = "models/best_seed_ood_classifier.pth"
    eval_results_path: str = "models/ood_evaluation_results.json"
    device: str = "auto"
    batch_size: int = 32
    num_workers: int = 4
    odin_temperature: float = 1000.0
    odin_epsilon: float = 0.0014
    ood_threshold: float = 0.2591
    max_image_size: int = 4096  # 最大图像尺寸限制
    timeout_seconds: int = 30   # 单张图像处理超时
    enable_performance_monitoring: bool = True
    log_level: str = "INFO"

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

@dataclass
class BatchResult:
    """批量处理结果"""
    total_images: int
    successful_images: int
    failed_images: int
    id_samples: int
    ood_samples: int
    avg_processing_time: float
    total_processing_time: float
    results: List[DetectionResult]
    
    def to_dict(self) -> Dict:
        """转换为字典格式"""
        return {
            'summary': {
                'total_images': self.total_images,
                'successful_images': self.successful_images,
                'failed_images': self.failed_images,
                'id_samples': self.id_samples,
                'ood_samples': self.ood_samples,
                'ood_rate': self.ood_samples / self.total_images if self.total_images > 0 else 0,
                'success_rate': self.successful_images / self.total_images if self.total_images > 0 else 0,
                'avg_processing_time': self.avg_processing_time,
                'total_processing_time': self.total_processing_time
            },
            'detailed_results': [result.to_dict() for result in self.results]
        }

class PerformanceMonitor:
    """性能监控器"""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """重置监控数据"""
        self.start_time = time.time()
        self.gpu_usage = []
        self.memory_usage = []
        self.processing_times = []
    
    def record_processing_time(self, processing_time: float):
        """记录处理时间"""
        self.processing_times.append(processing_time)
    
    def record_system_stats(self):
        """记录系统状态"""
        # CPU和内存使用率
        memory_percent = psutil.virtual_memory().percent
        self.memory_usage.append(memory_percent)
        
        # GPU使用率
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu_load = gpus[0].load * 100
                self.gpu_usage.append(gpu_load)
        except:
            pass
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        total_time = time.time() - self.start_time
        
        stats = {
            'total_time': total_time,
            'num_processed': len(self.processing_times),
            'avg_processing_time': np.mean(self.processing_times) if self.processing_times else 0,
            'min_processing_time': np.min(self.processing_times) if self.processing_times else 0,
            'max_processing_time': np.max(self.processing_times) if self.processing_times else 0,
            'throughput': len(self.processing_times) / total_time if total_time > 0 else 0
        }
        
        if self.memory_usage:
            stats['avg_memory_usage'] = np.mean(self.memory_usage)
            stats['max_memory_usage'] = np.max(self.memory_usage)
        
        if self.gpu_usage:
            stats['avg_gpu_usage'] = np.mean(self.gpu_usage)
            stats['max_gpu_usage'] = np.max(self.gpu_usage)
        
        return stats

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

class InferencePipeline:
    """完整的推理管道"""
    
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.performance_monitor = PerformanceMonitor() if config.enable_performance_monitoring else None
        
        # 设置日志级别
        logging.getLogger().setLevel(getattr(logging, config.log_level.upper()))
        
        # 设备设置
        if config.device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(config.device)
        
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
            with open(self.config.eval_results_path, 'r', encoding='utf-8') as f:
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
            checkpoint = torch.load(self.config.model_path, map_location=self.device)
            
            # 处理不同的checkpoint格式
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
            
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"模型加载完成: {self.config.model_path}")
            
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
    
    @contextmanager
    def _timeout_context(self, seconds: int):
        """超时上下文管理器"""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"处理超时 ({seconds}秒)")
        
        # 设置信号处理器
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(seconds)
        
        try:
            yield
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)
    
    def _validate_image(self, image_path: str) -> bool:
        """验证图像文件"""
        try:
            # 检查文件存在
            if not Path(image_path).exists():
                return False
            
            # 检查文件大小
            file_size = Path(image_path).stat().st_size
            if file_size > self.config.max_image_size * 1024 * 1024:  # 转换为字节
                logger.warning(f"图像文件过大: {image_path} ({file_size / 1024 / 1024:.1f}MB)")
                return False
            
            # 尝试打开图像
            with Image.open(image_path) as img:
                img.verify()
            
            return True
            
        except Exception as e:
            logger.warning(f"图像验证失败: {image_path}, 错误: {e}")
            return False
    
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
        perturbed_inputs = inputs - self.config.odin_epsilon * gradient
        perturbed_inputs = torch.clamp(perturbed_inputs, 0, 1)
        
        # 在扰动输入上计算温度缩放的logits
        with torch.no_grad():
            perturbed_logits = self.model(perturbed_inputs)
            scaled_logits = perturbed_logits / self.config.odin_temperature
            odin_scores = F.softmax(scaled_logits, dim=1).max(dim=1)[0]
        
        return logits, odin_scores
    
    def preprocess_image(self, image_path: str) -> torch.Tensor:
        """预处理图像"""
        # 验证图像
        if not self._validate_image(image_path):
            raise ValueError(f"图像验证失败: {image_path}")
        
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
            # 使用超时保护
            with self._timeout_context(self.config.timeout_seconds):
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
                    is_ood = odin_score < self.config.ood_threshold
                
                processing_time = time.time() - start_time
                
                # 记录性能
                if self.performance_monitor:
                    self.performance_monitor.record_processing_time(processing_time)
                    self.performance_monitor.record_system_stats()
                
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
    
    def detect_batch(self, image_paths: List[str], 
                    use_parallel: bool = True) -> BatchResult:
        """批量检测图像"""
        start_time = time.time()
        
        if self.performance_monitor:
            self.performance_monitor.reset()
        
        logger.info(f"开始批量检测 {len(image_paths)} 张图像")
        
        results = []
        
        if use_parallel and len(image_paths) > 1:
            # 并行处理
            max_workers = min(self.config.num_workers, len(image_paths))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_path = {
                    executor.submit(self.detect_single_image, path): path 
                    for path in image_paths
                }
                
                for i, future in enumerate(as_completed(future_to_path)):
                    result = future.result()
                    results.append(result)
                    
                    if (i + 1) % 10 == 0:
                        logger.info(f"已处理 {i + 1}/{len(image_paths)} 张图像")
        else:
            # 串行处理
            for i, image_path in enumerate(image_paths):
                result = self.detect_single_image(image_path)
                results.append(result)
                
                if (i + 1) % 10 == 0:
                    logger.info(f"已处理 {i + 1}/{len(image_paths)} 张图像")
        
        # 计算统计信息
        total_processing_time = time.time() - start_time
        successful_results = [r for r in results if r.error is None]
        failed_results = [r for r in results if r.error is not None]
        
        id_samples = sum(1 for r in successful_results if not r.is_ood)
        ood_samples = sum(1 for r in successful_results if r.is_ood)
        
        avg_processing_time = (
            np.mean([r.processing_time for r in successful_results]) 
            if successful_results else 0
        )
        
        batch_result = BatchResult(
            total_images=len(image_paths),
            successful_images=len(successful_results),
            failed_images=len(failed_results),
            id_samples=id_samples,
            ood_samples=ood_samples,
            avg_processing_time=avg_processing_time,
            total_processing_time=total_processing_time,
            results=results
        )
        
        logger.info(f"批量检测完成，共处理 {len(results)} 张图像")
        logger.info(f"成功: {len(successful_results)}, 失败: {len(failed_results)}")
        logger.info(f"ID样本: {id_samples}, OOD样本: {ood_samples}")
        
        return batch_result
    
    def save_results(self, batch_result: BatchResult, output_path: str):
        """保存检测结果"""
        try:
            # 添加性能监控信息
            result_dict = batch_result.to_dict()
            
            if self.performance_monitor:
                result_dict['performance_stats'] = self.performance_monitor.get_stats()
            
            result_dict['config'] = asdict(self.config)
            
            # 保存到文件
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, ensure_ascii=False, indent=2)
            
            logger.info(f"检测结果已保存到: {output_path}")
            
        except Exception as e:
            logger.error(f"保存结果失败: {e}")
            raise
    
    def print_summary(self, batch_result: BatchResult):
        """打印检测结果摘要"""
        print(f"\n🔍 推理管道检测结果摘要:")
        print(f"  总图像数: {batch_result.total_images}")
        print(f"  成功处理: {batch_result.successful_images} ({batch_result.successful_images/batch_result.total_images:.1%})")
        print(f"  处理失败: {batch_result.failed_images} ({batch_result.failed_images/batch_result.total_images:.1%})")
        print(f"  ID样本: {batch_result.id_samples} ({batch_result.id_samples/batch_result.total_images:.1%})")
        print(f"  OOD样本: {batch_result.ood_samples} ({batch_result.ood_samples/batch_result.total_images:.1%})")
        print(f"  平均处理时间: {batch_result.avg_processing_time:.3f}秒/图像")
        print(f"  总处理时间: {batch_result.total_processing_time:.2f}秒")
        print(f"  吞吐量: {batch_result.total_images/batch_result.total_processing_time:.1f}图像/秒")
        
        if self.performance_monitor:
            stats = self.performance_monitor.get_stats()
            print(f"\n📊 性能统计:")
            print(f"  处理吞吐量: {stats['throughput']:.1f}图像/秒")
            if 'avg_memory_usage' in stats:
                print(f"  平均内存使用: {stats['avg_memory_usage']:.1f}%")
            if 'avg_gpu_usage' in stats:
                print(f"  平均GPU使用: {stats['avg_gpu_usage']:.1f}%")

def load_config(config_path: str) -> InferenceConfig:
    """从文件加载配置"""
    if Path(config_path).exists():
        with open(config_path, 'r', encoding='utf-8') as f:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                config_dict = yaml.safe_load(f)
            else:
                config_dict = json.load(f)
        
        return InferenceConfig(**config_dict)
    else:
        logger.warning(f"配置文件不存在: {config_path}，使用默认配置")
        return InferenceConfig()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="种子分类和OOD检测推理管道")
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--input", type=str, required=True, help="输入图像路径或目录")
    parser.add_argument("--output", type=str, default="models/inference_results.json", help="输出结果文件路径")
    parser.add_argument("--parallel", action="store_true", help="启用并行处理")
    parser.add_argument("--device", type=str, choices=["auto", "cuda", "cpu"], help="指定设备")
    parser.add_argument("--batch-size", type=int, help="批处理大小")
    parser.add_argument("--threshold", type=float, help="OOD检测阈值")
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config:
        config = load_config(args.config)
    else:
        config = InferenceConfig()
    
    # 命令行参数覆盖配置
    if args.device:
        config.device = args.device
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.threshold:
        config.ood_threshold = args.threshold
    
    # 初始化推理管道
    pipeline = InferencePipeline(config)
    
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
    batch_result = pipeline.detect_batch(image_paths, use_parallel=args.parallel)
    
    # 打印摘要
    pipeline.print_summary(batch_result)
    
    # 保存结果
    pipeline.save_results(batch_result, args.output)
    
    # 打印一些详细结果
    print(f"\n📋 详细结果示例:")
    for i, result in enumerate(batch_result.results[:5]):
        if result.error:
            status = f"❌ 错误: {result.error}"
        else:
            status = "🚨 OOD" if result.is_ood else "✅ ID"
        
        print(f"  {i+1}. {result.filename}: {status} | "
              f"类别: {result.predicted_class_name} | "
              f"置信度: {result.confidence:.3f} | "
              f"时间: {result.processing_time:.3f}s")

if __name__ == "__main__":
    main() 