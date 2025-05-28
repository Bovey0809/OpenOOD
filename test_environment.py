#!/usr/bin/env python3
"""
环境测试脚本 - 验证 OpenOOD 环境配置
"""

import sys
import torch
import torchvision
import cv2
import numpy as np

def test_basic_imports():
    """测试基础库导入"""
    print("=== 基础库测试 ===")
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Torchvision version: {torchvision.__version__}")
    print(f"OpenCV version: {cv2.__version__}")
    print(f"NumPy version: {np.__version__}")
    print("✅ 基础库导入成功")

def test_cuda():
    """测试 CUDA 功能"""
    print("\n=== CUDA 测试 ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        
        # 测试简单的 GPU 操作
        x = torch.randn(3, 3).cuda()
        y = torch.randn(3, 3).cuda()
        z = torch.mm(x, y)
        print(f"GPU tensor operation test: {z.shape}")
        print("✅ CUDA 功能正常")
    else:
        print("❌ CUDA 不可用")

def test_openood_core():
    """测试 OpenOOD 核心功能"""
    print("\n=== OpenOOD 核心测试 ===")
    try:
        import openood
        print("✅ OpenOOD 主模块导入成功")
        
        # 测试数据集模块
        from openood.datasets import get_dataloader
        print("✅ 数据集模块导入成功")
        
        # 测试网络模块
        from openood.networks import get_network
        print("✅ 网络模块导入成功")
        
        # 测试预处理器
        from openood.preprocessors import get_preprocessor
        print("✅ 预处理器模块导入成功")
        
        # 测试训练器
        from openood.trainers import get_trainer
        print("✅ 训练器模块导入成功")
        
    except ImportError as e:
        print(f"❌ OpenOOD 模块导入失败: {e}")

def test_postprocessors():
    """测试后处理器（OOD 检测方法）"""
    print("\n=== OOD 检测方法测试 ===")
    try:
        # 测试基础的 OOD 方法
        from openood.postprocessors.msp_postprocessor import MSPPostprocessor
        print("✅ MSP 后处理器导入成功")
        
        from openood.postprocessors.odin_postprocessor import ODINPostprocessor
        print("✅ ODIN 后处理器导入成功")
        
        from openood.postprocessors.mds_postprocessor import MDSPostprocessor
        print("✅ MDS 后处理器导入成功")
        
    except ImportError as e:
        print(f"⚠️  部分后处理器导入失败: {e}")
        print("这可能是由于可选依赖缺失，但不影响核心功能")

def test_simple_model():
    """测试简单模型创建和推理"""
    print("\n=== 模型测试 ===")
    try:
        # 创建一个简单的 ResNet 模型
        import torchvision.models as models
        model = models.resnet18(pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 10)  # 10类分类
        
        # 测试前向传播
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x)
        
        print(f"模型输出形状: {output.shape}")
        print("✅ 模型创建和推理测试成功")
        
        # 如果有 GPU，测试 GPU 推理
        if torch.cuda.is_available():
            model = model.cuda()
            x = x.cuda()
            with torch.no_grad():
                output = model(x)
            print("✅ GPU 模型推理测试成功")
            
    except Exception as e:
        print(f"❌ 模型测试失败: {e}")

def main():
    """主测试函数"""
    print("OpenOOD 环境配置测试")
    print("=" * 50)
    
    test_basic_imports()
    test_cuda()
    test_openood_core()
    test_postprocessors()
    test_simple_model()
    
    print("\n" + "=" * 50)
    print("环境测试完成！")
    print("如果看到 ✅ 标记，说明对应功能正常")
    print("如果看到 ⚠️ 标记，说明有可选功能缺失但不影响核心使用")
    print("如果看到 ❌ 标记，说明有重要功能异常，需要修复")

if __name__ == "__main__":
    main() 