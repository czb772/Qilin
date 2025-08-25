# VLM性能测试指南

本目录包含了用于测试VLM模型在Qilin数据集上性能的脚本。

## 文件说明

- `test_vlm_performance.py` - 完整的VLM性能测试脚本
- `test_vlm_quick.py` - 快速验证脚本，用于测试模型加载和基本功能
- `run_vlm_test.sh` - 运行测试的shell脚本
- `config/test_vlm_config.yaml` - 测试配置文件

## 快速开始

### 1. 快速验证模型

首先运行快速测试脚本，验证模型是否能正常加载：

```bash
cd baselines
python test_vlm_quick.py
```

这个脚本会：
- 加载Qwen2.5-VL-7B-Instruct模型
- 测试基本的图像和文本处理能力
- 验证模型推理功能

### 2. 运行完整性能测试

使用shell脚本运行完整的性能测试：

```bash
cd baselines
./run_vlm_test.sh
```

或者直接使用Python脚本：

```bash
python test_vlm_performance.py \
    --model_path /data1/Qilin/baselines/model/Qwen2.5-VL-7B-Instruct/ \
    --sample_num 50 \
    --output_dir vlm_test_results \
    --device cuda
```

## 参数说明

- `--model_path`: VLM模型路径
- `--sample_num`: 测试样本数量（默认100）
- `--output_dir`: 结果输出目录（默认vlm_test_results）
- `--device`: 使用的设备（默认cuda）

## 测试内容

### 交叉编码器测试
- 对每个查询-文档对进行相关性评分
- 使用模型生成0/1判断
- 计算MRR、Recall、Precision等指标

### 双向编码器测试
- 预编码所有候选文档
- 编码查询文本
- 计算余弦相似度进行排序
- 评估检索性能

## 输出结果

测试完成后，结果会保存在指定的输出目录中：

```
vlm_test_results/
├── performance_metrics.json  # 性能指标汇总
└── 控制台输出               # 详细的测试过程
```

## 性能指标

- **MRR@k**: Mean Reciprocal Rank at k
- **Recall@k**: 召回率@k
- **Precision@k**: 精确率@k

## 注意事项

1. **显存需求**: 7B模型需要较大显存，建议使用GPU
2. **测试时间**: 完整测试可能需要较长时间，可以调整sample_num
3. **候选数量**: 可以调整max_candidates来平衡精度和速度
4. **批处理**: 可以调整batch_size来优化显存使用

## 故障排除

### 模型加载失败
- 检查模型路径是否正确
- 确认模型文件完整性
- 检查显存是否足够

### 推理失败
- 检查输入格式是否正确
- 确认处理器配置
- 查看错误日志

### 性能问题
- 调整batch_size
- 减少sample_num
- 使用更小的候选数量

## 示例输出

```
=== VLM性能测试 ===
开始时间: Wed Aug 22 15:30:00 CST 2024

Loading VLM model from: /data1/Qilin/baselines/model/Qwen2.5-VL-7B-Instruct/
Model and data loaded successfully!

=== Testing Cross-Encoder Performance (Sample: 50) ===
Testing Cross-Encoder: 100%|██████████| 50/50 [02:30<00:00]

Cross-Encoder Performance:
  MRR@1: 0.3200
  MRR@3: 0.4500
  MRR@5: 0.5200
  MRR@10: 0.5800
  Recall@10: 0.6500
  Precision@10: 0.1200

=== Testing Dual-Encoder Performance (Sample: 50) ===
Pre-encoding candidate documents: 100%|██████████| 25/25 [01:45<00:00]
Testing Dual-Encoder: 100%|██████████| 50/50 [00:30<00:00]

Dual-Encoder Performance:
  MRR@1: 0.2800
  MRR@3: 0.4200
  MRR@5: 0.4800
  MRR@10: 0.5400
  Recall@10: 0.6000
  Precision@10: 0.1000

Results saved to: vlm_test_results
测试完成!
结束时间: Wed Aug 22 15:35:00 CST 2024
```
