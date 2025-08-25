#!/bin/bash

# VLM性能测试运行脚本

echo "=== VLM性能测试 ==="
echo "开始时间: $(date)"

# 设置模型路径
MODEL_PATH="/data1/Qilin/baselines/model/Qwen2.5-VL-7B-Instruct/"

# 检查模型是否存在
if [ ! -d "$MODEL_PATH" ]; then
    echo "错误: 模型路径不存在: $MODEL_PATH"
    echo "请检查模型路径是否正确"
    exit 1
fi

echo "使用模型: $MODEL_PATH"

# 运行测试
echo "开始运行VLM性能测试..."

python test_vlm_performance.py \
    --model_path "$MODEL_PATH" \
    --sample_num 20 \
    --output_dir vlm_test_results \
    --device cuda

echo "测试完成!"
echo "结束时间: $(date)"
echo "结果保存在: vlm_test_results/"
