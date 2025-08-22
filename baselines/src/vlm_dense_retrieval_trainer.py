"""
VLM双向编码器训练器 - 用于多模态密集检索训练
"""

import torch
import torch.nn as nn
from trainer import BaseTrainer
from vlm_dense_retrieval_model import VLMDenseRetrievalModel

class VLMDenseRetrievalTrainer(BaseTrainer):
    """
    VLM双向编码器训练器
    
    训练策略：
    1. 对比学习：InfoNCE损失
    2. 批内负样本：使用批次内的其他文档作为负样本
    3. 难负样本挖掘：可选的难负样本策略
    """
    
    def setup_model(self):
        """设置VLM双向编码器模型"""
        self._handle_previous_checkpoints()
        self.model = VLMDenseRetrievalModel(self.config)
        if self.accelerator.is_main_process:
            print("VLM Dense Retrieval Model initialized")
    
    def contrastive_loss(self, query_embeddings, doc_embeddings, temperature=0.07):
        """
        计算InfoNCE对比损失
        
        Args:
            query_embeddings: 查询嵌入 [batch_size, embedding_dim]
            doc_embeddings: 文档嵌入 [batch_size * (1 + neg_samples), embedding_dim]
            temperature: 温度参数
            
        Returns:
            torch.Tensor: 损失值
        """
        batch_size = query_embeddings.size(0)
        
        # 计算相似度矩阵
        similarities = self.model.compute_similarity(query_embeddings, doc_embeddings)
        similarities = similarities / temperature
        
        # 标签：每个查询对应的正样本索引
        # 假设文档按照 [pos_1, neg_1_1, ..., neg_1_k, pos_2, neg_2_1, ..., neg_2_k, ...] 排列
        num_docs_per_query = doc_embeddings.size(0) // batch_size
        labels = torch.arange(batch_size, device=similarities.device) * num_docs_per_query
        
        # 交叉熵损失
        loss = nn.CrossEntropyLoss()(similarities, labels)
        
        return loss
    
    def _train_step(self, batch):
        """训练步骤"""
        self.model.train()
        self.optimizer.zero_grad()
        
        # 获取输入
        query_inputs = batch['query_inputs']
        doc_inputs = batch['doc_inputs']
        
        # 前向传播
        outputs = self.model(
            query_inputs=query_inputs,
            doc_inputs=doc_inputs
        )
        
        # 计算损失
        loss = self.contrastive_loss(
            outputs['query_embeddings'],
            outputs['doc_embeddings']
        )
        
        # 反向传播
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss

