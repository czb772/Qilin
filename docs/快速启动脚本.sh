#!/bin/bash

echo "=== Qilin VLM搜索和推荐实验复现脚本 ==="
echo ""

# 检查是否在正确的目录
if [ ! -f "README.md" ]; then
    echo "错误：请在Qilin仓库根目录运行此脚本"
    exit 1
fi

# 创建必要的目录
echo "创建必要的目录..."
mkdir -p model
mkdir -p datasets/qilin
mkdir -p baselines/output

# 检查依赖是否安装
echo "检查Python依赖..."
cd baselines
if ! python -c "import transformers, torch, datasets, accelerate, peft" 2>/dev/null; then
    echo "安装Python依赖..."
    pip install -r requirements.txt
else
    echo "Python依赖已安装"
fi
cd ..

echo ""
echo "=== 数据准备 ==="
echo "请按照以下步骤准备数据："
echo ""
echo "1. 下载Qilin数据集："
echo "   - 访问: https://huggingface.co/datasets/THUIR/qilin"
echo "   - 下载并解压到 datasets/qilin/ 目录"
echo ""
echo "2. 下载图像资源："
echo "   - 访问: https://cloud.tsinghua.edu.cn/d/af72ab5dbba1460da6c0/"
echo "   - 下载并解压到 datasets/qilin/image/ 目录"
echo ""
echo "3. 下载预训练模型："
echo "   # 原始Qwen2-VL模型"
echo "   git clone https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct model/Qwen2-VL-2B-Instruct"
echo "   git clone https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct model/Qwen2-VL-7B-Instruct"
echo ""
echo "   # Qwen2.5-VL模型（用于替换实验）"
echo "   git clone https://huggingface.co/Qwen/Qwen2.5-VL-2B-Instruct model/Qwen2.5-VL-2B-Instruct"
echo "   git clone https://huggingface.co/Qwen/Qwen2.5-VL-7B-Instruct model/Qwen2.5-VL-7B-Instruct"
echo ""
echo "   # BERT模型"
echo "   git clone https://huggingface.co/google-bert/bert-base-chinese model/bert-base-chinese"
echo ""

# 检查数据是否准备完成
echo "检查数据准备状态..."
if [ ! -d "datasets/qilin" ] || [ -z "$(ls -A datasets/qilin 2>/dev/null)" ]; then
    echo "警告：datasets/qilin 目录为空或不存在"
    echo "请先下载数据集"
    exit 1
fi

if [ ! -d "model" ] || [ -z "$(ls -A model 2>/dev/null)" ]; then
    echo "警告：model 目录为空或不存在"
    echo "请先下载预训练模型"
    exit 1
fi

echo "数据准备检查完成！"
echo ""

echo "=== 实验运行命令 ==="
echo ""
echo "原始实验（使用Qwen2-VL）："
echo ""
echo "# 搜索任务 - Qwen2-VL-2B"
echo "cd baselines"
echo "sh scripts/run.sh --config config/search_vlm_config.yaml --cuda 0"
echo ""
echo "# 搜索任务 - Qwen2-VL-7B"
echo "sh scripts/run.sh --config config/search_vlm7B_config.yaml --cuda 0"
echo ""
echo "# 推荐任务 - Qwen2-VL-2B"
echo "sh scripts/run.sh --config config/recommendation_vlm_config.yaml --cuda 0"
echo ""
echo "# 推荐任务 - Qwen2-VL-7B"
echo "sh scripts/run.sh --config config/recommendation_vlm7B_config.yaml --cuda 0"
echo ""
echo "替换实验（使用Qwen2.5-VL）："
echo ""
echo "# 搜索任务 - Qwen2.5-VL-2B"
echo "sh scripts/run.sh --config config/search_vlm_qwen25_2b_config.yaml --cuda 0"
echo ""
echo "# 搜索任务 - Qwen2.5-VL-7B"
echo "sh scripts/run.sh --config config/search_vlm_qwen25_7b_config.yaml --cuda 0"
echo ""
echo "# 推荐任务 - Qwen2.5-VL-2B"
echo "sh scripts/run.sh --config config/recommendation_vlm_qwen25_2b_config.yaml --cuda 0"
echo ""
echo "# 推荐任务 - Qwen2.5-VL-7B"
echo "sh scripts/run.sh --config config/recommendation_vlm_qwen25_7b_config.yaml --cuda 0"
echo ""
echo "监控训练过程："
echo "sh scripts/tensorboard.sh"
echo "访问 http://localhost:6006 查看训练日志"
echo ""

echo "=== 重要提示 ==="
echo ""
echo "1. 确保有足够的GPU内存（推荐24GB+用于7B模型）"
echo "2. 如果遇到OOM错误，可以减小batch_size"
echo "3. 训练结果保存在对应的output目录中"
echo "4. 评估结果保存在eval_results目录中"
echo "5. 主要评估指标：MRR@10"
echo ""

echo "=== 故障排除 ==="
echo ""
echo "常见问题："
echo "- CUDA内存不足：减小batch_size或使用gradient_checkpointing"
echo "- 模型加载失败：检查模型路径和文件完整性"
echo "- 数据加载错误：检查数据集路径和格式"
echo ""

echo "脚本执行完成！请按照上述步骤准备数据并运行实验。"
