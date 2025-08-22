"""
VLM双向编码器模型 - 用于多模态密集检索

主要功能：
1. 双塔架构：查询编码器和文档编码器
2. 支持多模态输入（文本+图像）
3. 使用对比学习训练
4. 支持LoRA微调
"""

import torch
import torch.nn as nn
from transformers import AutoModelForImageTextToText, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import os
from utils import mean_token_pool

class VLMDenseRetrievalModel(nn.Module):
    """
    VLM双向编码器模型 - 用于多模态检索
    
    架构特点：
    1. 双塔架构：使用同一个VLM模型，但分别编码查询和文档
    2. 支持文本+图像的多模态输入
    3. 输出固定维度的向量表示
    4. 支持参数共享或独立的查询/文档编码器
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        
        # 是否使用独立的查询和文档编码器
        self.tie_encoders = self.model_config.get('tie_encoders', True)
        
        # 初始化处理器
        self.processor = AutoProcessor.from_pretrained(
            self.model_config['model_name_or_path'],
            max_pixels=100 * 28 * 28,
            trust_remote_code=True
        )
        
        # 初始化模型
        self._setup_models()
        
        # 投影层（可选）：将模型输出映射到固定维度
        hidden_size = self.query_encoder.config.hidden_size
        self.embedding_dim = self.model_config.get('embedding_dim', 768)
        
        if hidden_size != self.embedding_dim:
            self.query_projection = nn.Linear(hidden_size, self.embedding_dim)
            self.doc_projection = nn.Linear(hidden_size, self.embedding_dim)
        else:
            self.query_projection = nn.Identity()
            self.doc_projection = nn.Identity()
    
    def _setup_models(self):
        """设置查询和文档编码器"""
        # 查询编码器
        self.query_encoder = AutoModelForImageTextToText.from_pretrained(
            self.model_config['model_name_or_path'],
            trust_remote_code=True,
            load_in_4bit=self.model_config.get('load_in_4bit', False)
        )
        
        # 启用梯度检查点
        if self.model_config.get('gradient_checkpointing', True):
            self.query_encoder.gradient_checkpointing_enable()
            self.query_encoder.enable_input_require_grads()
        
        # 设置LoRA
        if self.model_config.get('use_lora', True):
            self._setup_lora(self.query_encoder, 'query')
        
        # 文档编码器
        if self.tie_encoders:
            # 共享参数
            self.doc_encoder = self.query_encoder
        else:
            # 独立的文档编码器
            self.doc_encoder = AutoModelForImageTextToText.from_pretrained(
                self.model_config['model_name_or_path'],
                trust_remote_code=True,
                load_in_4bit=self.model_config.get('load_in_4bit', False)
            )
            
            if self.model_config.get('gradient_checkpointing', True):
                self.doc_encoder.gradient_checkpointing_enable()
                self.doc_encoder.enable_input_require_grads()
            
            if self.model_config.get('use_lora', True):
                self._setup_lora(self.doc_encoder, 'doc')
    
    def _setup_lora(self, model, adapter_name):
        """设置LoRA适配器"""
        lora_path = os.path.join(
            self.model_config['lora_checkpoint_dir'], 
            f'{adapter_name}_adapter'
        )
        
        if os.path.exists(os.path.join(lora_path, 'adapter_config.json')):
            model.load_adapter(lora_path, adapter_name, is_trainable=True)
            print(f"Loaded {adapter_name} LoRA adapter from {lora_path}")
        else:
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model.add_adapter(peft_config, adapter_name)
            print(f'Added {adapter_name} LoRA adapter from init')
    
    def encode_queries(self, inputs):
        """
        编码查询
        
        Args:
            inputs: 处理后的查询输入（可能包含文本和图像）
            
        Returns:
            torch.Tensor: 查询的向量表示
        """
        outputs = self.query_encoder(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # 池化
        query_embeddings = mean_token_pool(
            last_hidden_states=hidden_states,
            attention_mask=inputs['attention_mask']
        )
        
        # 投影
        query_embeddings = self.query_projection(query_embeddings)
        
        # L2归一化
        query_embeddings = torch.nn.functional.normalize(query_embeddings, p=2, dim=-1)
        
        return query_embeddings
    
    def encode_documents(self, inputs):
        """
        编码文档
        
        Args:
            inputs: 处理后的文档输入（可能包含文本和图像）
            
        Returns:
            torch.Tensor: 文档的向量表示
        """
        outputs = self.doc_encoder(**inputs, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]
        
        # 池化
        doc_embeddings = mean_token_pool(
            last_hidden_states=hidden_states,
            attention_mask=inputs['attention_mask']
        )
        
        # 投影
        doc_embeddings = self.doc_projection(doc_embeddings)
        
        # L2归一化
        doc_embeddings = torch.nn.functional.normalize(doc_embeddings, p=2, dim=-1)
        
        return doc_embeddings
    
    def forward(self, query_inputs=None, doc_inputs=None):
        """
        前向传播
        
        Args:
            query_inputs: 查询输入
            doc_inputs: 文档输入
            
        Returns:
            dict: 包含查询和文档嵌入的字典
        """
        results = {}
        
        if query_inputs is not None:
            results['query_embeddings'] = self.encode_queries(query_inputs)
        
        if doc_inputs is not None:
            results['doc_embeddings'] = self.encode_documents(doc_inputs)
        
        return results
    
    def compute_similarity(self, query_embeddings, doc_embeddings):
        """
        计算查询和文档之间的相似度
        
        Args:
            query_embeddings: 查询嵌入 [batch_size, embedding_dim]
            doc_embeddings: 文档嵌入 [batch_size * (1 + neg_samples), embedding_dim]
            
        Returns:
            torch.Tensor: 相似度矩阵
        """
        return torch.matmul(query_embeddings, doc_embeddings.transpose(0, 1))
    
    def save_pretrained(self, save_path):
        """保存模型"""
        # 保存查询编码器
        query_path = os.path.join(save_path, 'query_encoder')
        self.query_encoder.save_pretrained(query_path)
        
        # 如果文档编码器是独立的，也保存它
        if not self.tie_encoders:
            doc_path = os.path.join(save_path, 'doc_encoder')
            self.doc_encoder.save_pretrained(doc_path)
        
        # 保存投影层
        projection_path = os.path.join(save_path, 'projections.pt')
        torch.save({
            'query_projection': self.query_projection.state_dict(),
            'doc_projection': self.doc_projection.state_dict(),
            'embedding_dim': self.embedding_dim
        }, projection_path)
