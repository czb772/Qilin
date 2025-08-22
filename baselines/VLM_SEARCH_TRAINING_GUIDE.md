# VLM搜索训练详细指南

## 概述

本仓库实现了基于视觉语言模型（VLM）的搜索重排序系统，主要用于训练模型对查询-文档对进行相关性评分。系统支持多模态输入（文本+图像），使用交叉编码器架构进行训练。

## 核心组件

### 1. 模型架构

#### VLM交叉编码器模型 (`VLMCrossEncoderModel`)
- **架构类型**: 单塔架构，联合编码查询-文档对
- **基础模型**: Qwen2-VL-7B-Instruct等视觉语言模型
- **输入**: 查询文本 + 文档内容（文本+图像）
- **输出**: 相关性分数（0-1之间的概率值）
- **训练目标**: 二分类任务，区分相关和不相关的查询-文档对

#### 密集检索模型 (`DenseRetrievalModel`)
- **架构类型**: 双塔架构，分别编码查询和文档
- **用途**: 对比学习训练
- **输出**: 文档表示向量
- **训练目标**: 让相关查询-文档对的相似度更高

### 2. 数据处理流程

#### 训练数据处理器 (`VLMCrossEncoderTrainingDataProcessor`)
```python
# 数据加载
- 加载Qilin数据集（查询、文档、点击数据）
- 处理多模态数据（文本+图像）

# 样本构建
- 正样本：用户点击的文档
- 负样本：未点击的文档
- 图像处理：加载、预处理、拼接

# 输入格式
- 对话格式：用户问题 + 文档内容
- 多模态：文本 + 图像
```

#### 数据流程
1. **数据加载**: 从HuggingFace数据集加载Qilin数据
2. **样本构建**: 根据点击行为构建正负样本
3. **图像处理**: 加载文档相关图像，处理为模型输入格式
4. **文本处理**: 使用tokenizer处理文本输入
5. **批处理**: 组织成训练批次

### 3. 训练流程

#### 训练器 (`VLMCrossEncoderTrainer`)
```python
# 初始化阶段
1. 设置训练环境（Accelerator、分布式训练）
2. 加载预训练模型和tokenizer
3. 配置LoRA微调
4. 初始化优化器和学习率调度器
5. 加载训练数据

# 训练循环
for epoch in range(num_epochs):
    for batch in train_dataloader:
        # 前向传播
        logits = model(**inputs)
        
        # 计算损失
        loss = BCEWithLogitsLoss(logits, labels)
        
        # 反向传播
        accelerator.backward(loss)
        
        # 参数更新
        optimizer.step()
        scheduler.step()
        
        # 定期评估和保存
        if step % eval_steps == 0:
            evaluate()
        if step % save_steps == 0:
            save_checkpoint()
```

#### 训练策略
- **损失函数**: BCEWithLogitsLoss（二分类交叉熵损失）
- **优化器**: AdamW
- **学习率调度**: LinearLR（线性调度）
- **微调策略**: LoRA（参数高效微调）
- **显存优化**: 梯度检查点、4bit量化

### 4. 配置系统

#### 主要配置项
```yaml
# 模型配置
model:
  model_name_or_path: "Qwen2-VL-7B-Instruct"  # 基础模型
  user_lora: true                             # 使用LoRA微调
  gradient_checkpointing: true                # 梯度检查点

# 数据配置
datasets:
  batch_size: 32                              # 批次大小
  max_length: 512                             # 最大序列长度
  negative_samples: 1                         # 负样本数量

# 训练配置
training:
  num_epochs: 10000                           # 训练轮数
  eval_steps: 500                             # 评估频率
  save_steps: 500                             # 保存频率

# 优化器配置
optimizer:
  name: AdamW
  lr: 1e-4                                    # 学习率
  weight_decay: 0.01                          # 权重衰减
```

### 5. 启动流程

#### 单机训练
```bash
# 启动命令
sh scripts/run.sh --config config/search_vlm_config.yaml --cuda 1

# 执行流程
1. 解析命令行参数
2. 设置CUDA_VISIBLE_DEVICES
3. 创建输出目录
4. 配置分布式训练参数
5. 启动accelerate训练
```

#### 多机训练
```bash
# 主节点
sh scripts/run.sh --config config/search_vlm_config.yaml --cuda 0,1 --num_machines 2 --rank 0

# 从节点
sh scripts/run.sh --config config/search_vlm_config.yaml --cuda 0,1 --num_machines 2 --rank 1
```

### 6. 评估系统

#### 评估指标
- **MRR@10**: 平均倒数排名（主要指标）
- **NDCG@10**: 归一化折损累积增益
- **Recall@10**: 召回率

#### 评估流程
1. 加载训练好的模型
2. 对测试集进行推理
3. 计算相关性分数
4. 重排序搜索结果
5. 计算评估指标

### 7. 关键技术点

#### LoRA微调
```python
# LoRA配置
peft_config = LoraConfig(
    lora_alpha=32,      # 缩放参数
    lora_dropout=0.1,   # Dropout率
    r=16,               # 秩
    bias='none',        # 不训练偏置
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)
```

#### 多模态处理
```python
# 图像处理
- 加载原始图像
- 调整尺寸为1024x1024
- 与文本拼接为多模态输入
- 使用VLM模型处理多模态数据
```

#### 分布式训练
```python
# Accelerate配置
- 支持单机多卡
- 支持多机多卡
- 自动处理数据并行
- 集成DeepSpeed优化
```

### 8. 常见问题解决

#### 显存不足
1. 减小batch_size
2. 启用gradient_checkpointing
3. 使用load_in_4bit
4. 调整max_length

#### 训练不收敛
1. 调整学习率
2. 检查数据质量
3. 调整负样本数量
4. 检查损失函数

#### GPU选择问题
1. 正确设置--cuda参数
2. 检查CUDA_VISIBLE_DEVICES
3. 确保GPU可用性

### 9. 性能优化建议

#### 训练效率
- 使用更大的batch_size（显存允许）
- 启用混合精度训练
- 使用梯度累积
- 优化数据加载（num_workers）

#### 模型效果
- 调整负样本采样策略
- 使用更好的预训练模型
- 增加训练数据
- 调整模型超参数

### 10. 部署和使用

#### 模型保存
- LoRA适配器保存在lora_checkpoint_dir
- 分类器参数单独保存
- 支持断点续训

#### 推理使用
```python
# 加载模型
model = VLMCrossEncoderModel(config)
model.load_pretrained(checkpoint_path)

# 推理
inputs = prepare_inputs(query, document)
score = model(**inputs)
```

## 总结

本VLM搜索训练系统提供了一个完整的端到端解决方案，支持多模态搜索重排序任务。通过LoRA微调和分布式训练，可以在有限的计算资源下高效训练大型VLM模型。系统具有良好的可扩展性和可配置性，适用于不同的搜索场景。
