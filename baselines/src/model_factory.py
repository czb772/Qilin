"""
VLM搜索模型工厂 - 定义和实现各种用于搜索任务的模型

主要功能：
1. 定义基础模型类，提供通用的模型初始化和管理功能
2. 实现密集检索模型（DenseRetrievalModel）- 用于对比学习
3. 实现VLM交叉编码器模型（VLMCrossEncoderModel）- 用于重排序
4. 支持LoRA微调，减少显存占用
5. 集成BM25检索器作为基线方法

模型架构：
- 密集检索模型：双塔架构，分别编码查询和文档
- VLM交叉编码器：单塔架构，联合编码查询-文档对
- 支持多模态输入（文本+图像）
- 使用LoRA进行参数高效微调
"""

from utils import *
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from registry import register_class
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoProcessor, AutoModelForImageTextToText
from transformers import AutoConfig
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import os
import jieba
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
import torch
import torch.nn as nn

class BaseModel:
    """
    基础模型类 - 提供通用的模型初始化和管理功能
    
    主要功能：
    1. 加载预训练模型和tokenizer
    2. 配置模型参数（梯度检查点、LoRA等）
    3. 冻结非交叉注意力参数
    4. 提供模型保存和加载功能
    """
    def __init__(self, config):
        """
        初始化基础模型
        
        Args:
            config (dict): 模型配置字典
        """
        self.model_config = config['model']
        # 加载HuggingFace模型配置
        self.hf_model_config = AutoConfig.from_pretrained(self.model_config['model_name_or_path'])
        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config['tokenizer_name_or_path'],
            trust_remote_code=True
        )
        
        # 将配置中的参数应用到模型配置
        for key in self.model_config:
            self.hf_model_config.__dict__[key] = self.model_config[key]
            
        self.is_bert = 'bert' in self.model_config['model_name_or_path']
        self.model = self._create_model()
        
        # 启用梯度检查点以节省显存
        if self.model_config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
            
        # 对于非BERT模型，尝试冻结非交叉注意力参数
        if not self.is_bert:
            try:
                self._freeze_non_crossattention_parameters()
            except:
                print("freeze_non_crossattention_parameters failed")

class DenseRetrievalModel(BaseModel):
    """
    密集检索模型 - 用于对比学习训练
    
    主要特点：
    1. 双塔架构：分别编码查询和文档
    2. 支持LoRA微调
    3. 使用平均池化获取文档表示
    4. 适用于对比学习训练策略
    
    训练目标：
    - 让相关查询-文档对的相似度更高
    - 让不相关查询-文档对的相似度更低
    """
    def __init__(self, config):
        super().__init__(config)
        self.embedding_model = self.model

    def _create_model(self):
        """
        创建模型实例
        
        Returns:
            torch.nn.Module: 模型实例
        """
        if self.is_bert:
            # BERT类模型
            model = AutoModel.from_pretrained(
                self.model_config['model_name_or_path'],
                config=self.hf_model_config,
                trust_remote_code=True
            )
        else:
            # 因果语言模型（如Qwen2-VL）
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config['model_name_or_path'],
                config=self.hf_model_config,
                attn_implementation='eager',
                trust_remote_code=True
            )
            # 冻结输入嵌入层
            model.base_model.get_input_embeddings().weight.requires_grad = False
            
            # 设置LoRA
            self._setup_lora(model)
                
        return model

    def _setup_lora(self, model):
        """
        设置LoRA微调
        
        Args:
            model: 要添加LoRA的模型
        """
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        if os.path.exists(os.path.join(self.model_config['lora_checkpoint_dir'], 'adapter_config.json')):
            # 加载已有的LoRA适配器
            model.load_adapter(self.model_config['lora_checkpoint_dir'], 'retrieval')
            print("Load retrieval lora adapter from", self.model_config['lora_checkpoint_dir'])
        else:
            # 创建新的LoRA适配器
            peft_config = LoraConfig(
                lora_alpha=32,      # LoRA缩放参数
                lora_dropout=0.1,   # LoRA dropout率
                r=16,               # LoRA秩
                bias='none',        # 不训练偏置项
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],  # 目标模块
            )
            model.add_adapter(peft_config, "retrieval")
            print('Add retrieval lora adapter from init')

    def _freeze_non_crossattention_parameters(self):
        """冻结非交叉注意力参数以节省显存"""
        freeze_non_crossattention_parameters(self.model, True, True)

    def save_pretrained(self, save_path):
        """保存模型和tokenizer"""
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def forward(self, **inputs):
        """
        前向传播
        
        Args:
            **inputs: 模型输入（tokenized文本）
            
        Returns:
            torch.Tensor: 文档表示向量
        """
        outputs = self.model(**inputs)

        last_hidden_states = outputs.last_hidden_state
        
        # 使用平均token池化获取文档表示
        pooled_output = mean_token_pool(
            last_hidden_states=last_hidden_states,
            attention_mask=inputs['attention_mask']
        )
        
        return pooled_output

class CrossNetwork(nn.Module):
    def __init__(self, input_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        self.weights = nn.ParameterList([
            nn.Parameter(torch.randn(input_dim))
            for _ in range(num_layers)
        ])
        self.bias = nn.ParameterList([
            nn.Parameter(torch.zeros(input_dim))
            for _ in range(num_layers)
        ])
        # BatchNorm1d
        self.batch_norms = nn.ModuleList([
            nn.BatchNorm1d(input_dim)
            for _ in range(num_layers)
        ])
    
    def forward(self, x):
        x0 = x
        xi = x
        for i in range(self.num_layers):
            xw = torch.sum(xi * self.weights[i], dim=-1, keepdim=True)
            xi = x0 * xw + self.bias[i] + xi
            xi = self.batch_norms[i](xi)
        return xi

class DCNModel(nn.Module):
    def __init__(
        self,
        config,
        num_cross_layers=3,
        hidden_size=256,
        dropout_rate=0.2,
        user_id_embedding_dim=32,  
        query_from_type_num=16  
    ):
        super().__init__()

        self.query_from_type_num = query_from_type_num  
        self.text_hidden_size = 768
        
        # Sparse feature embedding layers
        self.user_embedding_dims = {
            'gender': (3, 8),          # Gender: 3 classes
            'platform': (4, 8),        # Platform: 4 classes
            'age': (10, 16),           # Age: 10 classes
            'location': (1096, 32),    # Geographic location
            'user_idx': (15483, user_id_embedding_dim),  # User ID: 0-15482
        }
        
        self.note_embedding_dims = {
            'note_type': (8, 8),       # Note type
            'taxonomy1_id': (43, 16),   # Level 1 category
            'taxonomy2_id': (311, 32),  # Level 2 category
            'taxonomy3_id': (548, 64),  # Level 3 category
            'note_idx': (1983940, user_id_embedding_dim)   # Note ID: 0-1983939
        }
        
        # Create embedding layers
        self.user_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.user_embedding_dims.items()
        })
        
        self.note_embeddings = nn.ModuleDict({
            k: nn.Embedding(dim[0], dim[1])
            for k, dim in self.note_embedding_dims.items()
        })
                
        # Calculate feature dimensions
        self.query_dim = self.text_hidden_size + query_from_type_num  # query_text + query_from_type
        self.user_dense_dim = 42  # User dense feature dimension
        self.note_dense_dim = 36  # Note dense feature dimension
        
        self.user_sparse_dim = sum(dim[1] for dim in self.user_embedding_dims.values())
        self.note_sparse_dim = sum(dim[1] for dim in self.note_embedding_dims.values())
        
        # Total feature dimension
        self.total_feature_dim = (
            self.query_dim +
            self.user_dense_dim +
            self.user_sparse_dim +
            self.note_dense_dim +
            self.note_sparse_dim +
            self.text_hidden_size +
            user_id_embedding_dim
        )
        
        # DCN layer
        self.cross_network = CrossNetwork(self.total_feature_dim, num_cross_layers)
        
        # DNN layer
        self.dnn = nn.Sequential(
            nn.Linear(self.total_feature_dim, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        
        # Output layer
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_size + self.total_feature_dim, 1),
            nn.BatchNorm1d(1)
        )
        
    def forward(self, query_features, user_features, note_features):
        # 1. Process query features
        query_text = query_features['question_embedding']
        query_features['query_from_type'] = query_features['query_from_type'].long()
        query_from_type = nn.functional.one_hot(query_features['query_from_type'], num_classes=self.query_from_type_num).float()
        query_vector = torch.cat([query_text, query_from_type], dim=-1)
        
        # 2. Process user features
        user_dense = user_features['dense']
        user_sparse_embeddings = []
        for key, embedding_layer in self.user_embeddings.items():
            user_sparse_embeddings.append(embedding_layer(user_features[key]))
        user_sparse = torch.cat(user_sparse_embeddings, dim=-1)
        
        # Process historical behavior - mean pooling
        history_embeddings = self.note_embeddings['note_idx'](user_features['recent_clicked_note_idxs'])
        history_encoding = torch.mean(history_embeddings, dim=1)
        
        # 3. Process note features
        note_text = note_features['note_embedding']
        note_dense = note_features['dense']
        note_sparse_embeddings = []
        for key, embedding_layer in self.note_embeddings.items():
            note_sparse_embeddings.append(embedding_layer(note_features[key]))
        note_sparse = torch.cat(note_sparse_embeddings, dim=-1)
        
        # 4. Concatenate features
        combined_features = torch.cat([
            query_vector,
            user_dense,
            user_sparse,
            history_encoding,
            note_dense,
            note_sparse,
            note_text
        ], dim=-1)
        
        # 5. DCN processing
        cross_output = self.cross_network(combined_features)
        dnn_output = self.dnn(combined_features)
        
        # 6. Merge DCN and DNN outputs
        final_output = torch.cat([cross_output, dnn_output], dim=-1)
        
        # 7. Output layer
        logits = self.output_layer(final_output)
        return logits.squeeze(-1)
    
    def get_loss(self, query_features, user_features, note_features, labels):
        predictions = self(query_features, user_features, note_features)
        return torch.nn.BCEWithLogitsLoss()(predictions, labels)
        
    def load_model(self, model_path):
        """Load model parameters from specified path
        
        Args:
            model_path (str): Path to model parameter file
        """            
        try:
            state_dict = torch.load(model_path, map_location='cpu')
            self.load_state_dict(state_dict, strict=False)
            print(f"Successfully loaded model parameters from {model_path}")
        except Exception as e:
            print(f"Error loading model parameters: {str(e)}")
            print('Reinitializing model parameters')

class CrossEncoderModel(torch.nn.Module, BaseModel):
    def __init__(self, config):
        torch.nn.Module.__init__(self)
        hf_config = AutoConfig.from_pretrained(config['model']['model_name_or_path'])
        self.classifier = torch.nn.Linear(hf_config.hidden_size, 1)
        BaseModel.__init__(self, config)

    def _create_model(self):
        if self.is_bert:
            model = AutoModel.from_pretrained(
                self.model_config['model_name_or_path'],
                config=self.hf_model_config,
                trust_remote_code=True
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_config['model_name_or_path'],
                config=self.hf_model_config,
                attn_implementation='eager',
                trust_remote_code=True
            )
            model.base_model.get_input_embeddings().weight.requires_grad = False
            
            self._setup_lora(model)
        
        classifier_path = os.path.join(self.model_config['model_name_or_path'], 'classifier.pt')
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path))
            print(f"Loaded classifier parameters from {classifier_path}")
                
        return model

    def _setup_lora(self, model):
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        if os.path.exists(os.path.join(self.model_config['lora_checkpoint_dir'], 'adapter_config.json')):
            model.load_adapter(self.model_config['lora_checkpoint_dir'], 'cross_encoder')
            print("Load cross_encoder lora adapter from", self.model_config['lora_checkpoint_dir'])
        else:
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model.add_adapter(peft_config, "cross_encoder")
            print('Add cross_encoder lora adapter from init')

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        classifier_path = os.path.join(save_path, 'classifier.pt')
        torch.save(self.classifier.state_dict(), classifier_path)

    def forward(self, **inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        
        cls_output = last_hidden_states[:, 0, :]
        logits = self.classifier(cls_output)
        
        return logits


class VLMCrossEncoderModel(torch.nn.Module):
    """
    VLM交叉编码器模型 - 用于搜索重排序任务
    
    主要特点：
    1. 单塔架构：联合编码查询-文档对
    2. 多模态输入：支持文本和图像
    3. 二分类输出：预测查询-文档的相关性
    4. 支持LoRA微调
    5. 使用线性分类器进行最终预测
    
    模型架构：
    - 基础模型：Qwen2-VL等视觉语言模型
    - 特征提取：使用最后一层的隐藏状态
    - 池化策略：平均token池化
    - 分类器：线性层 + sigmoid激活
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model_config = config['model']
        
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_config['model_name_or_path'],
            trust_remote_code=True,
            load_in_4bit=self.model_config['load_in_4bit']
        )
        if self.model_config['gradient_checkpointing']:
            self.model.gradient_checkpointing_enable()
            self.model.enable_input_require_grads()
        if self.model_config['user_lora']:
            self._setup_lora(self.model)
        
        self.classifier = torch.nn.Linear(self.model.config.hidden_size, 1)
        
        classifier_path = os.path.join(self.model_config['lora_checkpoint_dir'], 'classifier.pt')
        if os.path.exists(classifier_path):
            self.classifier.load_state_dict(torch.load(classifier_path))
            print(f"Loaded classifier parameters from {classifier_path}")
    
    def _setup_lora(self, model):
        print(f"Try to load lora model from {self.model_config['lora_checkpoint_dir']}")
        if os.path.exists(os.path.join(self.model_config['lora_checkpoint_dir'], 'adapter_config.json')):
            model.load_adapter(self.model_config['lora_checkpoint_dir'], 'cross_encoder', is_trainable=True)
            print("Load cross_encoder lora adapter from", self.model_config['lora_checkpoint_dir'])
        else:
            peft_config = LoraConfig(
                lora_alpha=32,
                lora_dropout=0.1,
                r=16,
                bias='none',
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )
            model.add_adapter(peft_config, "cross_encoder")
            print('Add cross_encoder lora adapter from init')

    def forward(self, **inputs):   
        outputs = self.model(**inputs, output_hidden_states=True)
        
        hidden_states = outputs.hidden_states[-1]
        features = mean_token_pool(
            last_hidden_states=hidden_states,
            attention_mask=inputs['attention_mask']
        )
        
        features = features.to(self.classifier.weight.dtype)
        logits = self.classifier(features)
        
        return logits

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)
        classifier_path = os.path.join(save_path, 'classifier.pt')
        torch.save(self.classifier.state_dict(), classifier_path)

class BM25Retriever:
    def __init__(self, passages: List[str], tokenizer=None):
        """
        Initialize BM25 retriever
        Args:
            passages: Collection of documents
            tokenizer: Tokenizer function, defaults to None
        """
        self.passages = passages
        self.tokenizer = tokenizer if tokenizer else self._default_tokenizer
        
        # Tokenize all passages
        self.tokenized_passages = [self.tokenizer(p) for p in passages]
        
        # Initialize BM25 model
        self.bm25 = BM25Okapi(self.tokenized_passages)
    
    def _default_tokenizer(self, text: str) -> List[str]:
        """
        Default tokenizer
        Args:
            text: Input text
        Returns:
            List of tokens
        """
        # Use jieba for Chinese text
        if any('\u4e00' <= char <= '\u9fff' for char in text):
            return list(jieba.cut(text))
        # Split by space for English text
        return text.lower().split()
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve relevant passages
        Args:
            query: Query text
            top_k: Number of results to return
        Returns:
            List of tuples containing (passage text, relevance score)
        """
        # Tokenize the query
        tokenized_query = self.tokenizer(query)
        
        # Calculate scores for all passages
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get indices of top_k highest scoring results
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return results
        results = []
        for idx in top_indices:
            if scores[idx] > 0:  # Only return results with positive scores
                results.append((idx, self.passages[idx], scores[idx]))
        
        return results

