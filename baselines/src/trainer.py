"""
VLM搜索训练器 - 用于训练视觉语言模型进行搜索任务
主要功能：
1. 训练VLM模型理解查询和文档的语义关系
2. 支持对比学习，通过正负样本训练模型区分相关和不相关内容
3. 集成DeepSpeed进行分布式训练和内存优化
4. 支持LoRA微调，减少显存占用
"""

from accelerate import Accelerator
from evaluator import *
from dataset_factory import *
from utils import *
from tqdm import tqdm
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, AutoModel
import torch_optimizer as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR
from peft import LoraConfig, get_peft_model, PeftModel, TaskType
import datetime
import shutil
import sys
from registry import registry, register_class
import time
from glob import glob
from model_factory import *

# 优化器映射表 - 支持多种优化器选择
optimizer_class = {"AdamW": FusedAdam, "Lamb": optim.Lamb, "DeepSpeedCPUAdam": DeepSpeedCPUAdam}
# 学习率调度器映射表
scheduler_class = {"CosineAnnealingLR": CosineAnnealingLR, "LinearLR": LinearLR}

def dataset_class(class_name):
    """根据类名获取数据集类"""
    cls = registry.get_class(class_name)
    if cls:
        return cls
    else:
        raise ValueError(f"Class {class_name} not found")

class BaseTrainer:
    """
    基础训练器类 - 提供训练的基础框架
    包含环境设置、模型初始化、数据加载、优化器配置等通用功能
    """
    def __init__(self, config):
        """
        初始化训练器
        Args:
            config: 训练配置字典，包含模型、数据、训练等所有参数
        """
        self.config = config
        self.setup_environment()  # 设置训练环境（Accelerator、分布式等）
        self.setup_tracking()     # 设置指标跟踪
        self.setup_model()        # 初始化模型
        self.setup_data()         # 设置数据加载
        self.setup_optimization() # 设置优化器和调度器

    def setup_environment(self):
        """
        设置训练环境
        - 初始化Accelerator用于分布式训练
        - 设置日志记录和项目跟踪
        - 获取当前进程信息
        """
        self.accelerator = Accelerator(
            log_with=self.config['logger']['log_with'], 
            project_dir=self.config['project_dir']
        )
        if self.accelerator.is_main_process:
            print_args(self.config)
        self.accelerator.init_trackers(project_name=f'{self.config["project_name"]}')
        
        self.local_rank = self.accelerator.process_index  # 当前设备编号
        self.num_processes = self.accelerator.num_processes  # 总进程数
        self.step = 0  # 训练步数

    def setup_model(self):
        """初始化模型 - 由子类实现具体逻辑"""
        raise NotImplementedError

    def setup_data(self):
        """设置数据加载 - 由子类实现具体逻辑"""
        raise NotImplementedError

    def setup_optimization(self):
        """
        设置优化器和学习率调度器
        - 加载优化器配置
        - 加载学习率调度器
        - 准备训练环境
        """
        self.load_optimizer()
        self.load_scheduler()
        self.prepare_for_training()

    def setup_tracking(self):
        """设置指标跟踪 - 用于记录最佳模型"""
        self.target_metric = self.config['evaluation']['target_metric']  # 目标评估指标
        self.best_metric = -1  # 最佳指标值

    def load_optimizer(self):
        """
        加载优化器
        - 根据配置选择优化器类型（AdamW、Lamb等）
        - 只对需要梯度的参数进行优化
        """
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']
        # 只选择需要梯度的参数
        params = [(k, v) for k, v in self.model.model.named_parameters() if v.requires_grad]
        params = {'params': [v for k, v in params]}
        self.optimizer = optimizer_class[optimizer_name]([params], **optimizer_config['kwargs'])

    def load_scheduler(self):
        """
        加载学习率调度器
        - 支持余弦退火、线性调度等
        """
        scheduler_config = self.config['scheduler']
        scheduler_name = scheduler_config['name']
        self.scheduler = scheduler_class[scheduler_name](
            self.optimizer, 
            **scheduler_config['kwargs']
        )

    def prepare_for_training(self):
        """准备训练环境 - 由子类实现具体逻辑"""
        raise NotImplementedError

    def train(self):
        """
        主训练循环
        - 按epoch进行训练
        - 每个epoch结束后进行评估（可选）
        """
        # self.evaluate()
        for epoch in range(1, self.config['training']['num_epochs']):
            self.train_epoch(epoch)
            # self.evaluate()

    def train_epoch(self, epoch):
        """train one epoch"""
        raise NotImplementedError

    def evaluate(self):
        """evaluation - to be implemented by subclass"""
        raise NotImplementedError

    def save_checkpoint(self, suffix='', is_best=True):
        """save checkpoint - to be implemented by subclass"""
        raise NotImplementedError

    def _dist_gather_tensor(self, t):
        """gather tensors from all processes"""
        if t is None:
            return None
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.num_processes)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.local_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

class DenseRetrievalTrainer(BaseTrainer):
    """
    密集检索模型训练器 - 专门用于训练VLM进行搜索任务
    主要特点：
    1. 支持对比学习，通过正负样本训练模型
    2. 使用DeepSpeed进行分布式训练和内存优化
    3. 支持LoRA微调，减少显存占用
    4. 自动处理checkpoint的保存和加载
    """
    def setup_model(self):
        """
        设置模型
        - 处理之前的checkpoint
        - 初始化密集检索模型
        - 打印可训练参数统计
        """
        self._handle_previous_checkpoints()
        self.model = DenseRetrievalModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model.model)

    def setup_data(self):
        self.load_training_data()
        self.build_evaluator()
        self.negatives_x_device = self.config['training']['negatives_x_device']

    def prepare_for_training(self):
        self.model.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        """Handle previous checkpoints"""
        if self.accelerator.is_main_process:
            if self.config['model']['load_from_new']:
                self._load_latest_checkpoint()
            self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        """Load the best checkpoint"""
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            self._copy_checkpoint_files(best_file_path)

    def _find_best_checkpoint(self, file_paths):
        """find the best checkpoint"""
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path

    def _load_latest_checkpoint(self):
        """Load the latest checkpoint"""
        latest_dir = find_latest_dir_with_subdir(self.config['base_project_dir'])
        if latest_dir:
            self._copy_from_dir(latest_dir)

    def _copy_checkpoint_files(self, source_path):
        """Copy checkpoint files"""
        source_dir = os.path.dirname(source_path)
        self._copy_from_dir(source_dir)
        print(f'Best {self.target_metric} is {self.best_metric}')

    def _copy_from_dir(self, source_dir):
        """Copy files from specified directory"""
        for cand in ['base_lora_checkpoint_dir']:
            cand_path = self.config['model'][cand]
            if os.path.exists(f'{source_dir}/{cand_path}'):
                shutil.copytree(
                    f'{source_dir}/{cand_path}', 
                    f"{self.config['project_dir']}/{cand_path}"
                )
                self.config['model']['model_name_or_path'] = f"{self.config['project_dir']}/{cand_path}"

    def build_evaluator(self):
        """build evaluator"""
        self.evaluation_config = self.config['evaluation']
        if self.evaluation_config['evaluate_type'] == 'rerank':
            self.test_loader = self._create_test_loader()
            self.evaluator = DenseRetrievalRerankingEvaluator(
                self.accelerator,
                self.model,
                self.test_loader,
                **self.evaluation_config
            )
        else:
            self.note_loader = self._create_note_loader()
            self.query_loader = self._create_query_loader()
            self.evaluator = DenseRetrievalEvaluator(
                self.accelerator,
                self.model,
                self.note_loader,
                self.query_loader,
                **self.evaluation_config
            )
    
    def _create_test_loader(self):
        return DenseRetrievalRerankingTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def _create_note_loader(self):
        return NoteDataProcessor(
            self.local_rank,
            self.num_processes,
            **self.config['datasets']
        ).get_dataloader()

    def _create_query_loader(self):
        return QueryDataProcessor(
            self.local_rank,
            self.num_processes,
            **self.config['datasets']
        ).get_dataloader()

    def load_training_data(self):
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)
                
        pbar.close()

    def _train_step(self, batch):
        self.model.model.train()
        self.optimizer.zero_grad()
        
        query_emb, passage_emb = self._get_embeddings(batch)
        loss = self.contrastive_loss(query_emb, passage_emb)
        
        self.accelerator.backward(loss)
        self.optimizer.step()
        self.scheduler.step()
        
        return loss

    def _get_embeddings(self, batch):
        """get the embeddings"""
        if self.config['model']['tie_model_weights']:
            return self._get_tied_embeddings(batch)
        else:
            raise NotImplementedError

    def _get_tied_embeddings(self, batch):
        batch_size = batch['queries_tokenized']['input_ids'].shape[0]
        merged_tokenized = batch['merged_tokenized']
        merged_emb = self.model.forward(**merged_tokenized)
        query_emb = merged_emb[:batch_size, :]
        passage_emb = merged_emb[batch_size:, :]
        return query_emb, passage_emb

    def _update_progress(self, pbar, epoch, step, loss):
        self.step += 1
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _handle_periodic_actions(self, loss, epoch, step):
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}
        
        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            self.evaluate()
            
        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)
            
        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def contrastive_loss(self, query_emb, passage_emb):
        cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='mean')
        
        if self.negatives_x_device:
            query_emb = self._dist_gather_tensor(query_emb)
            passage_emb = self._dist_gather_tensor(passage_emb)
            
        scores = torch.matmul(query_emb, passage_emb.transpose(0, 1))
        scores = scores.view(query_emb.size(0), -1)
        
        labels = torch.arange(
            scores.size(0), 
            device=scores.device, 
            dtype=torch.long
        )
        labels = labels * (passage_emb.size(0) // query_emb.size(0))
        
        return cross_entropy_loss(scores, labels)

    def evaluate(self):
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def save_checkpoint(self, suffix='', is_best=True):
        save_paths = self._get_save_paths(suffix)
        model = self.accelerator.unwrap_model(self.model.model)
        model.save_pretrained(save_paths['lora'])

        if is_best:
            self._save_best_metric(save_paths['project'])

    def _get_save_paths(self, suffix):
        base_paths = {
            'lora': self.config['model']['lora_checkpoint_dir'],
            'project': self.config['project_dir']
        }
        
        save_paths = {}
        for key, base_path in base_paths.items():
            if key != 'project':
                save_paths[key] = os.path.join(base_path, suffix) if suffix else base_path
                os.makedirs(save_paths[key], exist_ok=True)
        save_paths['project'] = base_paths['project']
        
        return save_paths

    def _save_best_metric(self, project_path):
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)

class DCNTrainer(BaseTrainer):
    """DCN Model Trainer for Search"""
    def __init__(self, config):
        super().__init__(config)
        self.grad_stats = {
            'max_grad': 0.0,
            'min_grad': float('inf'),
            'grad_norm_history': [],
            'gradient_vanishing_count': 0,
            'gradient_exploding_count': 0
        }
        # Set thresholds for gradient vanishing and exploding
        self.grad_vanish_threshold = 1e-4
        self.grad_explode_threshold = 10.0

    def load_optimizer(self):
        """Load optimizer"""
        optimizer_config = self.config['optimizer']
        optimizer_name = optimizer_config['name']
        params = [(k, v) for k, v in self.model.named_parameters() if v.requires_grad]
        non_embedding_params = {'params': [v for k, v in params if 'embedding' not in k]}
        embedding_params = {'params': [v for k, v in params if 'embedding' in k], 'lr':1e-1}
        self.optimizer = optimizer_class[optimizer_name]([non_embedding_params, embedding_params], **optimizer_config['kwargs'])

    def _check_gradients(self):
        """Check gradient status, including vanishing and exploding"""
        total_norm = 0.0
        max_grad = 0.0
        min_grad = float('inf')
        has_valid_grad = False
        no_grad_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
                
            if param.grad is None:
                no_grad_params.append(name)
                continue
                
            param_norm = param.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
            
            # Get max and min values of non-zero gradients
            grad_abs = param.grad.data.abs()
            grad_nonzero = grad_abs[grad_abs > 0]
            if grad_nonzero.numel() > 0:
                has_valid_grad = True
                max_grad = max(max_grad, grad_nonzero.max().item())
                min_grad = min(min_grad, grad_nonzero.min().item())
            
            # Check gradient vanishing
            if param_norm < self.grad_vanish_threshold:
                self.grad_stats['gradient_vanishing_count'] += 1
                if self.accelerator.is_local_main_process:
                    print(f"\nWarning: Parameter {name} may have vanishing gradient (norm: {param_norm:.6f})")
            
            # Check gradient exploding
            if param_norm > self.grad_explode_threshold:
                self.grad_stats['gradient_exploding_count'] += 1
                if self.accelerator.is_local_main_process:
                    print(f"\nWarning: Parameter {name} may have exploding gradient (norm: {param_norm:.6f})")
            
            # Record detailed gradient information
            if self.accelerator.is_local_main_process and self.step % 100 == 0:
                print(f"Gradient statistics for parameter {name}:\n"
                      f"  - Norm: {param_norm:.6f}\n"
                      f"  - Max value: {grad_abs.max().item():.6f}\n"
                      f"  - Min non-zero value: {grad_nonzero.min().item() if grad_nonzero.numel() > 0 else 0:.6f}\n"
                      f"  - Zero gradient ratio: {(grad_abs == 0).float().mean().item():.2%}")
        
        # Print warning if parameters have no gradients
        if no_grad_params and self.accelerator.is_local_main_process:
            print(f"\nWarning: The following parameters have no gradients:\n{', '.join(no_grad_params)}")
        
        total_norm = total_norm ** 0.5
        self.grad_stats['max_grad'] = max(self.grad_stats['max_grad'], max_grad)
        if has_valid_grad:
            self.grad_stats['min_grad'] = min(self.grad_stats['min_grad'], min_grad)
        
        return total_norm

    def _log_gradient_stats(self):
        """Log gradient statistics"""
        if self.accelerator.is_local_main_process:
            stats = {
                'gradient/max_grad': self.grad_stats['max_grad'],
                'gradient/min_grad': self.grad_stats['min_grad'],
                'gradient/vanishing_count': self.grad_stats['gradient_vanishing_count'],
                'gradient/exploding_count': self.grad_stats['gradient_exploding_count']
            }
            
            if len(self.grad_stats['grad_norm_history']) > 0:
                stats['gradient/mean_norm'] = sum(self.grad_stats['grad_norm_history']) / len(self.grad_stats['grad_norm_history'])
            
            self.accelerator.log(stats, step=self.step)

    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = DCNModel(
                    self.config,
                    num_cross_layers=3,
                    hidden_size=256*2,
                    dropout_rate=0.1,
                    user_id_embedding_dim=32*2  
                )
        model_path = os.path.join(self.config['model']['model_name_or_path'], 'dcn_model.pt')
        print(f'loading model from {model_path}')
        self.model.load_model(model_path)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)

    def setup_data(self):
        self.load_training_data()
        self.build_evaluator()

    def prepare_for_training(self):
        self.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        if self.config['model']['load_from_new']:
            self._load_latest_checkpoint()
        self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            self._copy_checkpoint_files(best_file_path)

    def _find_best_checkpoint(self, file_paths):
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path

    def _load_latest_checkpoint(self):
        latest_dir = find_latest_dir_with_subdir(self.config['base_project_dir'])
        if latest_dir:
            self._copy_from_dir(latest_dir)

    def _copy_checkpoint_files(self, source_path):
        source_dir = os.path.dirname(source_path)
        self._copy_from_dir(source_dir)
        print(f'Best {self.target_metric} is {self.best_metric}')

    def _copy_from_dir(self, source_dir):
        for cand in ['base_lora_checkpoint_dir']:
            cand_path = self.config['model'][cand]
            print(f'copying {cand_path} from {source_dir} to {self.config["project_dir"]}')
            if os.path.exists(f'{source_dir}/{cand_path}'):
                if self.accelerator.is_main_process:
                    shutil.copytree(
                        f'{source_dir}/{cand_path}', 
                        f"{self.config['project_dir']}/{cand_path}"
                    )
                self.config['model']['model_name_or_path'] = f"{self.config['project_dir']}/{cand_path}"

    def build_evaluator(self):
        self.test_loader = self._create_test_loader()
        self.evaluation_config = self.config['evaluation']
        self.evaluator = DCNEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        return DCNTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def load_training_data(self):
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)
                
        pbar.close()

    def _train_step(self, batch):
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        query_features, user_features, note_features, labels = batch
        device = self.accelerator.device        
        query_features = {k: v.to(device) for k, v in query_features.items()}
        user_features = {k: v.to(device) for k, v in user_features.items()}
        note_features = {k: v.to(device) for k, v in note_features.items()}
        labels = labels.to(device)
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                print(f"Warning: Parameter {name} does not require gradients")
        criterion = torch.nn.BCEWithLogitsLoss()
        logits = self.model(query_features, user_features, note_features)
        loss = criterion(logits.squeeze(-1), labels)     

        self.accelerator.backward(loss)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()        
        return loss

    def _update_progress(self, pbar, epoch, step, loss):
        self.step += 1
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _handle_periodic_actions(self, loss, epoch, step):
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}
        
        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            self.evaluate()
            
        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)
            
        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def evaluate(self):
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def save_checkpoint(self, suffix='', is_best=True):
        save_paths = self._get_save_paths(suffix)
        model = self.accelerator.unwrap_model(self.model)
        torch.save(model.state_dict(), os.path.join(save_paths['lora'], 'dcn_model.pt'))

        if is_best:
            self._save_best_metric(save_paths['project'])

    def _get_save_paths(self, suffix):
        base_paths = {
            'lora': self.config['model']['lora_checkpoint_dir'],
            'project': self.config['project_dir']
        }
        
        save_paths = {}
        for key, base_path in base_paths.items():
            if key != 'project':
                save_paths[key] = os.path.join(base_path, suffix) if suffix else base_path
                os.makedirs(save_paths[key], exist_ok=True)
        save_paths['project'] = base_paths['project']
        
        return save_paths

    def _save_best_metric(self, project_path):
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)

class CrossEncoderTrainer(BaseTrainer):
    """Cross-encoder model trainer"""
    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = CrossEncoderModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model.model)

    def setup_data(self):
        self.load_training_data()
        self.build_evaluator()

    def prepare_for_training(self):
        self.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        """Handle previous checkpoints"""
        if self.accelerator.is_main_process:
            if self.config['model']['load_from_new']:
                self._load_latest_checkpoint()
            self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        """Load the best checkpoint"""
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            self._copy_checkpoint_files(best_file_path)

    def _find_best_checkpoint(self, file_paths):
        """Find the best checkpoint"""
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path

    def _load_latest_checkpoint(self):
        """Load the latest checkpoint"""
        latest_dir = find_latest_dir_with_subdir(self.config['base_project_dir'])
        if latest_dir:
            self._copy_from_dir(latest_dir)

    def _copy_checkpoint_files(self, source_path):
        """Copy checkpoint files"""
        source_dir = os.path.dirname(source_path)
        self._copy_from_dir(source_dir)
        print(f'Best {self.target_metric} is {self.best_metric}')

    def _copy_from_dir(self, source_dir):
        """Copy files from specified directory"""
        for cand in ['base_lora_checkpoint_dir']:
            cand_path = self.config['model'][cand]
            if os.path.exists(f'{source_dir}/{cand_path}'):
                shutil.copytree(
                    f'{source_dir}/{cand_path}', 
                    f"{self.config['project_dir']}/{cand_path}"
                )
                self.config['model']['model_name_or_path'] = f"{self.config['project_dir']}/{cand_path}"

    def build_evaluator(self):
        """Build evaluator"""
        self.test_loader = self._create_test_loader()
        self.evaluation_config = self.config['evaluation']
        self.evaluator = CrossEncoderEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        """Create test data loader"""
        return CrossEncoderTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def load_training_data(self):
        """Load training data"""
        dataset_config = self.config['datasets']
        train_data_processor = dataset_class(dataset_config['train_data_processor'])
        self.train_data_loader = train_data_processor(**dataset_config).get_dataloader()
        self.accelerator.wait_for_everyone()

    def train_epoch(self, epoch):
        """Train one epoch"""
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)
                
        pbar.close()

    def _train_step(self, batch):
        """Train one step"""
        self.model.train()
        inputs = batch['inputs']
        labels = batch['labels']
        
        # Move data to the correct device
        inputs = {k: v.to(self.accelerator.device) for k, v in inputs.items()}
        labels = labels.to(self.accelerator.device)
        
        # Use automatic mixed precision
        # Forward pass
        logits = self.model(**inputs)
        
        # Calculate loss
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.view(-1), labels.view(-1))
        
        # Backward pass
        self.accelerator.backward(loss)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss

    def _update_progress(self, pbar, epoch, step, loss):
        """Update progress bar and step count"""
        self.step += 1
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _handle_periodic_actions(self, loss, epoch, step):
        """Handle periodic operations"""
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}
        
        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            self.evaluate()
            
        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)
            
        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def evaluate(self):
        """Evaluate model"""
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)

    def _log_metrics(self, metrics):
        """Log evaluation metrics"""
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()

    def save_checkpoint(self, suffix='', is_best=True):
        """Save checkpoint"""
        save_paths = self._get_save_paths(suffix)
        model = self.accelerator.unwrap_model(self.model)
        model.save_pretrained(save_paths['lora'])

        if is_best:
            self._save_best_metric(save_paths['project'])

    def _get_save_paths(self, suffix):
        """Get save paths"""
        base_paths = {
            'lora': self.config['model']['lora_checkpoint_dir'],
            'project': self.config['project_dir']
        }
        
        save_paths = {}
        for key, base_path in base_paths.items():
            if key != 'project':
                save_paths[key] = os.path.join(base_path, suffix) if suffix else base_path
                os.makedirs(save_paths[key], exist_ok=True)
        save_paths['project'] = base_paths['project']

        return save_paths

    def _save_best_metric(self, project_path):
        """Save best metric"""
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        # Retrieval results directory
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)


class VLMCrossEncoderTrainer(BaseTrainer):
    """
    VLM交叉编码器训练器 - 专门用于训练视觉语言模型进行搜索重排序任务
    
    主要功能：
    1. 训练VLM模型作为交叉编码器，对查询-文档对进行相关性评分
    2. 使用二分类损失函数（BCEWithLogitsLoss）训练模型
    3. 支持LoRA微调，减少显存占用
    4. 自动处理checkpoint的保存和加载
    5. 集成DeepSpeed进行分布式训练
    
    训练流程：
    1. 输入：查询文本 + 文档内容（可能包含图像）
    2. 模型：VLM交叉编码器，输出相关性分数
    3. 损失：二分类交叉熵损失
    4. 优化：AdamW优化器 + 学习率调度器
    """
    def setup_model(self):
        self._handle_previous_checkpoints()
        self.model = VLMCrossEncoderModel(self.config)
        if self.accelerator.is_main_process:
            print_trainable_params_stats(self.model)

    def setup_data(self):
        """Set up data loading and evaluator"""
        self.load_training_data()
        self.build_evaluator()

    def prepare_for_training(self):
        """Prepare training environment"""
        self.model, self.optimizer, self.train_data_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, 
                self.optimizer, 
                self.train_data_loader, 
                self.scheduler
            )

    def _handle_previous_checkpoints(self):
        """Handle previous checkpoints"""
        if self.accelerator.is_main_process:
            self._load_best_checkpoint()
        self.accelerator.wait_for_everyone()

    def _load_best_checkpoint(self):
        """Load the best checkpoint"""
        base_project_dir = self.config['base_project_dir']
        result_file_paths = glob(base_project_dir+f'/*/best_{self.target_metric}.txt')
        best_file_path = self._find_best_checkpoint(result_file_paths)
        if best_file_path:
            self._copy_checkpoint_files(best_file_path)
    
    def _find_best_checkpoint(self, file_paths):
        """Find the best checkpoint"""
        best_file_path = ''
        for file_path in file_paths:
            score = float(open(file_path).readline())
            print(f'{file_path} {score}')
            if score > self.best_metric:
                self.best_metric = score
                best_file_path = file_path
        return best_file_path
    
    def _copy_checkpoint_files(self, source_path):
        """Copy checkpoint files"""
        source_dir = os.path.dirname(source_path)
        self._copy_from_dir(source_dir)
        print(f'Best {self.target_metric} is {self.best_metric}')

    def _copy_from_dir(self, source_dir):
        """Copy files from specified directory"""
        for cand in ['base_lora_checkpoint_dir']:
            cand_path = self.config['model'][cand]
            if os.path.exists(f'{source_dir}/{cand_path}'):
                if self.accelerator.is_main_process:
                    shutil.copytree(
                        f'{source_dir}/{cand_path}',
                        f"{self.config['project_dir']}/{cand_path}"
                    )
                self.config['model']['lora_checkpoint_dir'] = f"{self.config['project_dir']}/{cand_path}"

    def load_training_data(self):
        """Load training data"""
        dataset_config = self.config['datasets']
        train_data_processor = VLMCrossEncoderTrainingDataProcessor(**dataset_config)
        self.train_data_loader = train_data_processor.get_dataloader()
        self.accelerator.wait_for_everyone()

    def build_evaluator(self):
        """Build evaluator"""
        self.test_loader = self._create_test_loader()
        self.evaluation_config = self.config['evaluation']
        self.evaluator = VLMCrossEncoderEvaluator(
            self.accelerator,
            self.model,
            self.test_loader,
            **self.evaluation_config
        )

    def _create_test_loader(self):
        """Create test data loader"""
        accelerator = self.accelerator
        return VLMCrossEncoderTestDataProcessor(
            local_rank=self.local_rank,
            num_processes=self.num_processes,
            **self.config['datasets'],
            **self.config['evaluation']
        ).get_dataloader()

    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        pbar = tqdm(
            total=len(self.train_data_loader), 
            disable=not self.accelerator.is_local_main_process
        )
        
        for step, batch in enumerate(self.train_data_loader):
            loss = self._train_step(batch)
            self._update_progress(pbar, epoch, step, loss)
            self._handle_periodic_actions(loss, epoch, step)
                
        pbar.close()

    def _train_step(self, batch):
        """Train one step"""
        self.model.train()
        self.accelerator.unwrap_model(self.model).model.enable_input_require_grads()
        inputs = batch['inputs']
        labels = batch['labels']
        
        # Move data to the correct device
        inputs = {
            k: v.to(self.accelerator.device) 
            for k, v in inputs.items() 
            if isinstance(v, torch.Tensor)
        }
        labels = labels.to(self.accelerator.device)
        

        logits = self.model(**inputs)
        
        loss_fn = torch.nn.BCEWithLogitsLoss()
        loss = loss_fn(logits.view(-1), labels.view(-1))
        
        self.accelerator.backward(loss)
        
        self.optimizer.step()
        self.scheduler.step()
        self.optimizer.zero_grad()
        
        return loss

    def save_checkpoint(self, suffix='', is_best=True):
        """Save checkpoint"""
        save_paths = self._get_save_paths(suffix)
        
        model = self.accelerator.unwrap_model(self.model)
        
        model.save_pretrained(save_paths['lora'])
            
        if is_best:
            self._save_best_metric(save_paths['project'])

    def _get_save_paths(self, suffix):
        base_paths = {
            'lora': self.config['model']['lora_checkpoint_dir'],
            'project': self.config['project_dir']
        }

        save_paths = {}
        for key, base_path in base_paths.items():
            if key!= 'project':
                save_paths[key] = os.path.join(base_path, suffix) if suffix else base_path
                os.makedirs(save_paths[key], exist_ok=True)
        save_paths['project'] = base_paths['project']

        return save_paths
    
    def _save_best_metric(self, project_path):
        """save the results of best metric"""
        metric_path = os.path.join(project_path, f'best_{self.target_metric}.txt')
        with open(metric_path, 'w') as f:
            f.write(str(self.best_metric))
        result_dir = self.config['evaluation']['output_dir']
        target_dir = os.path.join(project_path, 'best_results')
        os.makedirs(target_dir, exist_ok=True)
        shutil.copytree(result_dir, target_dir, dirs_exist_ok=True)


    def _update_progress(self, pbar, epoch, step, loss):
        """Update progress bar and step count"""
        self.step += 1
        pbar.update(1)
        pbar.set_description(
            f"Epoch {epoch} - Step {step} - Loss {loss.cpu().detach().float().numpy():.4f}"
        )

    def _log_training_info(self, epoch, step, loss):
        """Log training information"""
        if self.accelerator.is_local_main_process:
            info = {
                'epoch': epoch,
                'step': step,
                'loss': loss.item(),
                'learning_rate': self.scheduler.get_last_lr()[0]
            }
            self.accelerator.log(info, step=self.step)

    def _handle_periodic_actions(self, loss, epoch, step):
        """Handle periodic operations"""
        stats = {'training/loss': float(loss.cpu().detach().float().numpy())}
        if self.step % self.config['training']['eval_steps'] == 0 or (epoch % self.config['training']['eval_epochs'] == 0 and step==0):
            self.evaluate()

        if self.step % self.config['training']['save_steps'] == 0 or epoch % self.config['training']['save_epochs'] == 0:
            self.save_checkpoint(suffix="new", is_best=False)

        if self.accelerator.is_local_main_process:
            self.accelerator.log(stats, step=self.step)

    def _log_metrics(self, metrics):
        """Log evaluation metrics"""
        for key, val in metrics.items():
            self.accelerator.log({f'evaluation/{key}': val}, step=self.step)
            if self.target_metric == key and val > self.best_metric:
                self.best_metric = val
                self.save_checkpoint()
    
    def evaluate(self):
        """Evaluate model"""
        metrics = self.evaluator.evaluate()
        if self.accelerator.is_local_main_process:
            self._log_metrics(metrics)


trainer_class = {
    'cross_encoder_trainer': CrossEncoderTrainer,
    'dense_retrieval_trainer': DenseRetrievalTrainer,
    'dcn_trainer': DCNTrainer,
    'vlm_trainer': VLMCrossEncoderTrainer,
}

if __name__=="__main__":
    config_path = sys.argv[1]
    config = get_config(config_path)
    time_stamp = sys.argv[2]
    if len(sys.argv) > 3:
        machine_rank = int(sys.argv[3])
        num_machines = int(sys.argv[4])
    else:
        machine_rank = 0
        num_machines = 1
    config['evaluation']['machine_rank'] = machine_rank
    config['evaluation']['num_machines'] = num_machines
    config['base_project_dir'] = config['project_dir']
    config['project_dir'] = os.path.join(config['project_dir'], f"{time_stamp}")
    project_dir = config['project_dir']
    config['evaluation']['output_dir'] = os.path.join(project_dir, config['evaluation']['output_dir'])
    if config['model']['load_from_new']:
        config['model']['lora_checkpoint_dir'] = os.path.join(config['model']['lora_checkpoint_dir'], 'new')
    config['model']['base_lora_checkpoint_dir'] = config['model']['lora_checkpoint_dir']
    config['model']['lora_checkpoint_dir'] = os.path.join(project_dir, config['model']['lora_checkpoint_dir'])
    config['optimizer']['kwargs']['lr'] = float(config['optimizer']['kwargs']['lr'])
    config['optimizer']['kwargs']['eps'] = float(config['optimizer']['kwargs']['eps'])
    trainer = trainer_class[config['trainer']](config)
    trainer.train()

