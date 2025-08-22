#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Qwen2.5-VL模型兼容性检查脚本
用于验证Qwen2.5-VL模型是否能够正常加载和使用
"""

import os
import sys
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
from PIL import Image
import numpy as np

def check_model_compatibility(model_path):
    """检查模型兼容性"""
    print(f"检查模型: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"错误: 模型路径不存在: {model_path}")
        return False
    
    try:
        # 检查模型文件
        required_files = ['config.json', 'pytorch_model.bin', 'tokenizer.json']
        for file in required_files:
            file_path = os.path.join(model_path, file)
            if not os.path.exists(file_path):
                print(f"警告: 缺少文件 {file}")
        
        # 尝试加载tokenizer
        print("加载tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        print("✓ Tokenizer加载成功")
        
        # 尝试加载processor
        print("加载processor...")
        processor = AutoProcessor.from_pretrained(
            model_path,
            max_pixels=100 * 28 * 28,
            trust_remote_code=True
        )
        print("✓ Processor加载成功")
        
        # 尝试加载模型（使用4bit量化以节省内存）
        print("加载模型...")
        model = AutoModelForImageTextToText.from_pretrained(
            model_path,
            trust_remote_code=True,
            load_in_4bit=True,
            device_map="auto"
        )
        print("✓ 模型加载成功")
        
        # 测试基本功能
        print("测试基本功能...")
        
        # 创建测试图像
        test_image = Image.new('RGB', (224, 224), color='white')
        
        # 测试文本
        test_text = "这是一张图片"
        
        # 使用processor处理输入
        inputs = processor(
            text=test_text,
            images=test_image,
            return_tensors="pt"
        )
        
        # 移动到GPU（如果可用）
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            model = model.cuda()
        
        # 测试前向传播
        with torch.no_grad():
            outputs = model(**inputs)
        
        print("✓ 前向传播测试成功")
        
        # 测试生成
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7
            )
        
        # 解码输出
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        print(f"✓ 生成测试成功，输出: {generated_text[:100]}...")
        
        print(f"✓ 模型 {model_path} 兼容性检查通过！")
        return True
        
    except Exception as e:
        print(f"✗ 模型兼容性检查失败: {str(e)}")
        return False

def main():
    """主函数"""
    print("=== Qwen2.5-VL模型兼容性检查 ===")
    print()
    
    # 检查的模型列表
    models_to_check = [
        "../model/Qwen2-VL-2B-Instruct",
        "../model/Qwen2-VL-7B-Instruct", 
        "../model/Qwen2.5-VL-2B-Instruct",
        "../model/Qwen2.5-VL-7B-Instruct"
    ]
    
    results = {}
    
    for model_path in models_to_check:
        print(f"\n{'='*50}")
        results[model_path] = check_model_compatibility(model_path)
        print(f"{'='*50}")
    
    # 总结结果
    print("\n=== 检查结果总结 ===")
    for model_path, success in results.items():
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{model_path}: {status}")
    
    # 给出建议
    print("\n=== 建议 ===")
    if results.get("../model/Qwen2.5-VL-2B-Instruct", False):
        print("✓ Qwen2.5-VL-2B-Instruct 可以正常使用")
    else:
        print("✗ Qwen2.5-VL-2B-Instruct 存在问题，请检查模型文件")
    
    if results.get("../model/Qwen2.5-VL-7B-Instruct", False):
        print("✓ Qwen2.5-VL-7B-Instruct 可以正常使用")
    else:
        print("✗ Qwen2.5-VL-7B-Instruct 存在问题，请检查模型文件")
    
    print("\n如果所有模型都检查通过，您可以开始运行实验了！")

if __name__ == "__main__":
    main()
