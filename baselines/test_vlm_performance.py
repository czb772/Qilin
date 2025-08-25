"""
VLM性能测试脚本 -  直接测试模型性能

功能：
1. 加载预训练VLM模型
2. 在Qilin测试集上评估性能
3. 支持交叉编码器和双向编码器测试
4. 输出详细的评估指标
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import os
from transformers import AutoModelForImageTextToText, AutoProcessor
from datasets import load_dataset
from PIL import Image
import json
from collections import defaultdict
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# 导入原始评估器的指标计算函数
def calculate_metrics(sorted_results, ground_truth, k_list):
    """
    Calculate MRR@k, MAP@k, Recall@k, Precision@k for multiple k values
    """
    max_k = max(k_list)  
    metrics = {k: {"mrr": 0.0, "map_sum": 0.0, "recall": 0.0, "precision": 0.0} for k in k_list}
    num_queries = len(ground_truth)
    valid_queries = 0

    for qid, relevant_pids in ground_truth.items():
        if qid not in sorted_results:
            num_queries -= 1  
            continue
        
        valid_queries += 1
        retrieved_pids = sorted_results[qid][:max_k]
        hits = [pid in relevant_pids for pid in retrieved_pids]

        # Calculate metrics for each k value
        for k in k_list:
            hits_at_k = hits[:k]
            
            # Calculate MRR@k
            if any(hits_at_k):
                first_hit_rank = hits_at_k.index(True) + 1  
                metrics[k]["mrr"] += 1 / first_hit_rank

            # Calculate MAP@k
            avg_precision = 0.0
            num_correct = 0
            for i, is_relevant in enumerate(hits_at_k):
                if is_relevant:
                    num_correct += 1
                    precision_at_i = num_correct / (i + 1)
                    avg_precision += precision_at_i
            if num_correct > 0:  
                avg_precision /= min(len(relevant_pids), k)  
                metrics[k]["map_sum"] += avg_precision

            # Calculate Recall@k
            metrics[k]["recall"] += sum(hits_at_k) / len(relevant_pids)

            # Calculate Precision@k
            metrics[k]["precision"] += sum(hits_at_k) / k

    # Return all zeros if no valid queries
    if valid_queries == 0:
        return {f"{metric}@{k}": 0.0 
                for k in k_list 
                for metric in ["MRR", "MAP", "Recall", "Precision"]}

    # Calculate average metrics for all k values
    results = {}
    for k in k_list:
        results[f"MRR@{k}"] = metrics[k]["mrr"] / valid_queries
        results[f"MAP@{k}"] = metrics[k]["map_sum"] / valid_queries
        results[f"Recall@{k}"] = metrics[k]["recall"] / valid_queries
        results[f"Precision@{k}"] = metrics[k]["precision"] / valid_queries

    return results

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

class VLMPerformanceTester:
    """VLM性能测试器"""
    
    def __init__(self, model_path, device='cuda'):
        self.device = device
        self.model_path = model_path.rstrip('/')
        
        print(f"Loading VLM model from: {self.model_path}")
        
        # 加载模型和处理器
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            torch_dtype=torch.float16,  # 使用半精度节省显存
            device_map='auto'
        )
        self.model.eval()
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )
        
        # 加载数据
        self.load_test_data()
        
        print("Model and data loaded successfully!")
    
    def load_test_data(self):
        """加载测试数据"""
        print("Loading Qilin test dataset...")
        
        # 加载文档语料库
        self.corpus = load_dataset("/data1/Qilin/datasets/qilin", 'notes')['train']
        
        # 加载测试查询
        self.test_data = load_dataset("/data1/Qilin/datasets/qilin", 'search_test')['train']
        
        # 加载相关性标注
        self.qrels = self.load_qrels("/data1/Qilin/datasets/search.test.qrels.csv")
        
        print(f"Loaded {len(self.test_data)} test queries")
        print(f"Loaded {len(self.corpus)} documents")
        print(f"Loaded {len(self.qrels)} relevance annotations")
    
    def load_qrels(self, qrels_path):
        """加载相关性标注"""
        df = pd.read_csv(qrels_path)
        qrels = defaultdict(list)
        for _, row in df.iterrows():
            qrels[int(row["qid"])].append(int(row["pid"]))
        return qrels
    
    def create_default_image(self):
        """创建默认图像"""
        return Image.new('RGB', (1024, 1024), color='white')
    
    def load_note_image(self, image_paths):
        """加载笔记图像"""
        if not image_paths:
            return self.create_default_image()
        try:
            image_path = os.path.join('/data1/Qilin/datasets', image_paths[0])
            image = Image.open(image_path)
            image = image.resize((1024, 1024))
            return image
        except Exception as e:
            print(f"Failed to load image: {e}")
            return self.create_default_image()
    
    def test_cross_encoder_performance(self, sample_num=100):
        """测试交叉编码器性能 - 参考原始VLM评估器"""
        print(f"\n=== Testing Cross-Encoder Performance (Sample: {sample_num}) ===")
        
        # 使用原始评估器的格式
        predictions = defaultdict(list)
        
        # 选择测试样本
        test_samples = self.test_data.select(range(min(sample_num, len(self.test_data))))
        
        for idx, item in enumerate(tqdm(test_samples, desc="Testing Cross-Encoder")):
            query = item["query"]
            search_results = item.get("search_results", [])
            
            if not search_results:
                continue
            
            # 获取候选文档
            candidates = search_results[:100]  # 限制候选数量
            
            # 计算相关性分数
            for note_idx in candidates:
                score = self.compute_cross_encoder_score(query, note_idx)
                # 使用idx作为qid，因为qrels中的qid是从0开始的连续数字
                qid = idx
                predictions[qid].append((note_idx, score))
        
        # 按原始评估器格式整理结果
        sorted_results = {}
        for qid, preds in predictions.items():
            sorted_results[qid] = [pid for pid, _ in sorted(preds, key=lambda x: x[1], reverse=True)]
        
        # 使用原始的指标计算函数
        metrics = calculate_metrics(sorted_results, self.qrels, [1, 3, 5, 10])
        
        print("\nCross-Encoder Performance:")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 50)
        
        return metrics, sorted_results
    
    def test_dual_encoder_performance(self, sample_num=100):
        """测试双向编码器性能"""
        print(f"\n=== Testing Dual-Encoder Performance (Sample: {sample_num}) ===")
        
        # 预编码所有候选文档
        print("Pre-encoding candidate documents...")
        doc_embeddings = {}
        doc_ids = set()
        
        # 收集所有候选文档ID
        test_samples = self.test_data.select(range(min(sample_num, len(self.test_data))))
        for item in test_samples:
            search_results = item.get("search_results", [])
            doc_ids.update(search_results[:100])
        
        # 批量编码文档
        batch_size = 8
        doc_list = list(doc_ids)
        
        # 单个处理文档，使用完整VLM功能
        for doc_idx in tqdm(doc_list, desc="Encoding documents"):
            try:
                # 使用完整的VLM编码文档
                embedding = self.encode_single_document(doc_idx)
                doc_embeddings[doc_idx] = embedding
            except Exception as e:
                print(f"Error encoding document {doc_idx}: {e}")
                # 如果编码失败，使用零向量
                doc_embeddings[doc_idx] = torch.zeros(4096).to(self.device)
        
        # 使用原始评估器的格式
        predictions = defaultdict(list)
        
        for idx, item in enumerate(tqdm(test_samples, desc="Testing Dual-Encoder")):
            query = item["query"]
            search_results = item.get("search_results", [])
            
            if not search_results:
                continue
            
            # 编码查询 - 使用完整VLM功能
            query_embedding = self.encode_query(query)
            
            # 获取候选文档并计算相似度
            candidates = search_results[:100]
            
            for note_idx in candidates:
                if note_idx in doc_embeddings:
                    # 计算余弦相似度
                    similarity = torch.cosine_similarity(
                        query_embedding.unsqueeze(0), 
                        doc_embeddings[note_idx].unsqueeze(0)
                    ).item()
                    
                    # 使用idx作为qid，因为qrels中的qid是从0开始的连续数字
                    qid = idx
                    predictions[qid].append((note_idx, similarity))
        
        # 按原始评估器格式整理结果
        sorted_results = {}
        for qid, preds in predictions.items():
            sorted_results[qid] = [pid for pid, _ in sorted(preds, key=lambda x: x[1], reverse=True)]
        
        # 调试：检查结果
        print(f"Dual-encoder sorted_results keys: {list(sorted_results.keys())[:5]}")
        print(f"Number of queries with results: {len(sorted_results)}")
        print(f"Qrels keys sample: {list(self.qrels.keys())[:5]}")
        print(f"Number of qrels: {len(self.qrels)}")
        
        # 使用原始的指标计算函数
        metrics = calculate_metrics(sorted_results, self.qrels, [1, 3, 5, 10])
        
        print("\nDual-Encoder Performance:")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 50)
        
        return metrics, sorted_results
    
    def test_dual_encoder_quick(self, sample_num=2, doc_limit=5):
        """快速测试双向编码器性能 - 只处理少量数据"""
        print(f"\n=== Quick Dual-Encoder Test (Sample: {sample_num}, Docs per query: {doc_limit}) ===")
        
        # 使用原始评估器的格式
        predictions = defaultdict(list)
        
        # 选择测试样本
        test_samples = self.test_data.select(range(min(sample_num, len(self.test_data))))
        
        for idx, item in enumerate(tqdm(test_samples, desc="Quick Dual-Encoder Test")):
            query = item["query"]
            search_results = item.get("search_results", [])
            
            if not search_results:
                continue
            
            print(f"\nQuery {idx}: {query[:]}...")
            
            # 只处理前几个候选文档
            candidates = search_results[:doc_limit]
            print(f"Processing {len(candidates)} documents...")
            
            # 编码查询
            query_embedding = self.encode_query(query)
            
            # 处理每个候选文档
            doc_details = []  # 存储文档详情用于显示
            for doc_idx, note_idx in enumerate(candidates):
                print(f"  Processing doc {doc_idx+1}/{len(candidates)}: {note_idx}")
                
                # 获取文档内容用于显示
                note = self.corpus[note_idx]
                doc_title = note['note_title']
                doc_content = note['note_content']
                doc_image_paths = note.get('image_path', [])
                
                # 编码文档
                doc_embedding = self.encode_single_document(note_idx)
                
                # 计算余弦相似度
                similarity = torch.cosine_similarity(
                    query_embedding.unsqueeze(0), 
                    doc_embedding.unsqueeze(0)
                ).item()
                
                print(f"    Similarity: {similarity:.4f}")
                print(f"    Title: {doc_title[:]}...")
                print(f"    Content: {doc_content[:]}...")
                if doc_image_paths:
                    print(f"    Images: {doc_image_paths}")
                else:
                    print(f"    Images: 无图片")
                
                doc_details.append({
                    'note_idx': note_idx,
                    'title': doc_title,
                    'content': doc_content,
                    'image_paths': doc_image_paths,
                    'similarity': similarity
                })
                
                qid = idx
                predictions[qid].append((note_idx, similarity))
            
            # 按相似度排序并显示
            doc_details.sort(key=lambda x: x['similarity'], reverse=True)
            # 显示真实相关文档（如果存在）
            qid = idx
            if qid in self.qrels:
                relevant_docs = self.qrels[qid]
                print(f"\n✅ Query {idx} 的真实相关文档: {relevant_docs}")
                
                # 显示真实相关文档的内容
                print("📖 真实相关文档内容:")
                for rel_doc_idx in relevant_docs[:]:
                    if rel_doc_idx < len(self.corpus):
                        rel_note = self.corpus[rel_doc_idx]
                        rel_image_paths = rel_note.get('image_path', [])
                        print(f"   📄 文档ID {rel_doc_idx}:")
                        print(f"      标题: {rel_note['note_title']}")
                        print(f"      内容: {rel_note['note_content'][:]}...")
                        if rel_image_paths:
                            print(f"      图片: {rel_image_paths}")
                        else:
                            print(f"      图片: 无图片")
                        print("      " + "="*80)
            else:
                print(f"\n❌ Query {idx} 在qrels中未找到相关文档")
            
            print(f"\n📋 Query {idx} VLM排序结果:")
            for rank, doc in enumerate(doc_details, 1):
                # 检查是否是真实相关文档
                is_relevant = "✅" if qid in self.qrels and doc['note_idx'] in self.qrels[qid] else "❌"
                print(f"  {rank}. {is_relevant} [相似度: {doc['similarity']:.4f}]")
                print(f"     标题: {doc['title']}")
                print(f"     内容: {doc['content'][:]}...")
                if doc['image_paths']:
                    print(f"     图片: {doc['image_paths']}")
                else:
                    print(f"     图片: 无图片")
                print(f"     文档ID: {doc['note_idx']}")
                print("     " + "-"*80)
        
        # 按原始评估器格式整理结果
        sorted_results = {}
        for qid, preds in predictions.items():
            sorted_results[qid] = [pid for pid, _ in sorted(preds, key=lambda x: x[1], reverse=True)]
            print(f"Query {qid} ranking: {sorted_results[qid]}")
        
        # 使用原始的指标计算函数
        metrics = calculate_metrics(sorted_results, self.qrels, [1, 3, 5])
        
        print("\nQuick Dual-Encoder Performance:")
        print("=" * 50)
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 50)
        
        return metrics, sorted_results

    def compute_cross_encoder_score(self, query, note_idx):
        """计算交叉编码器分数 - 使用模型隐藏状态输出分数而不是生成文本"""
        try:
            # 获取文档内容 - 参考原始数据处理器
            note = self.corpus[note_idx]
            doc_title = note['note_title']
            doc_content = note['note_content']
            doc_image = self.load_note_image(note.get('image_path', []))
            
            # 构建对话 - 参考原始VLM数据处理器
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{doc_title} {doc_content}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                ]
            }]
            
            # 使用原始方法：先生成文本提示，然后统一处理
            text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
            
            # 使用processor统一处理文本和图像 - 关键修复，禁用截断
            inputs = self.processor(
                text=[text_prompt],
                images=[doc_image],
                padding=True,
                return_tensors="pt"
            )
            
            # 移动到设备
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 使用模型直接输出相关性分数 - 参考原始VLM评估器
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # 使用最后一层隐藏状态的平均值作为相关性分数
                hidden_states = outputs.hidden_states[-1]
                score = hidden_states.mean().item()
                
                # 归一化到0-1范围
                normalized_score = torch.sigmoid(torch.tensor(score)).item()
            
            # 调试：打印分数
            print(f"Raw score for note {note_idx}: {score:.4f}, Normalized: {normalized_score:.4f}")
            
            return normalized_score
            
        except Exception as e:
            print(f"Error computing score for note {note_idx}: {e}")
            return 0.0
        
    
    def encode_query(self, query):
        """编码查询 - 使用原始VLM方法"""
        # 为查询创建默认图像以保持输入格式一致
        default_image = self.create_default_image()
        
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"查询：{query}"}
            ]
        }]
        
        # 使用原始方法：先生成文本提示，然后统一处理
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
        
        # 使用processor统一处理文本和图像，禁用截断
        inputs = self.processor(
            text=[text_prompt],
            images=[default_image],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            # 使用最后一层的平均池化
            hidden_states = outputs.hidden_states[-1]
            query_embedding = hidden_states.mean(dim=1)  # [1, hidden_size]
        
        return query_embedding.squeeze(0)
    
    def encode_query_simple(self, query):
        """简化查询编码"""
        try:
            inputs = self.processor.tokenizer(
                f"查询：{query}",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                query_embedding = hidden_states.mean(dim=1)
            
            return query_embedding.squeeze(0)
        except Exception as e:
            print(f"Error encoding query: {e}")
            return torch.zeros(4096).to(self.device)
    
    def encode_document_simple(self, note_idx):
        """简化文档编码"""
        try:
            note = self.corpus[note_idx]
            doc_title = note['note_title']
            doc_content = note['note_content']
            
            inputs = self.processor.tokenizer(
                f"文档：{doc_title} {doc_content}",
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.model(**inputs, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]
                doc_embedding = hidden_states.mean(dim=1)
            
            return doc_embedding.squeeze(0)
        except Exception as e:
            print(f"Error encoding document {note_idx}: {e}")
            return torch.zeros(4096).to(self.device)
    
    def encode_single_document(self, note_idx):
        """编码单个文档 - 使用原始VLM方法"""
        note = self.corpus[note_idx]
        doc_title = note['note_title']
        doc_content = note['note_content']
        doc_image = self.load_note_image(note.get('image_path', []))
        
        conversation = [{
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"文档标题：{doc_title}\n文档内容：{doc_content}"}
            ]
        }]
        
        # 使用原始方法：先生成文本提示，然后统一处理
        text_prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
        
        # 使用processor统一处理文本和图像，禁用截断
        inputs = self.processor(
            text=[text_prompt],
            images=[doc_image],
            padding=True,
            return_tensors="pt"
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            doc_embedding = hidden_states.mean(dim=1)  # [1, hidden_size]
        
        return doc_embedding.squeeze(0)
    
    def encode_documents_batch(self, doc_indices):
        """批量编码文档"""
        conversations = []
        images = []
        
        for note_idx in doc_indices:
            note = self.corpus[note_idx]
            doc_title = note['note_title']
            doc_content = note['note_content']
            doc_image = self.load_note_image(note.get('image_path', []))
            
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"文档标题：{doc_title}\n文档内容：{doc_content}"}
                ]
            }]
            
            conversations.append(conversation)
            images.append(doc_image)
        
        # 批量处理 - 添加padding
        inputs = self.processor.apply_chat_template(
            conversations, 
            tokenize=True, 
            add_generation_prompt=False,
            return_tensors="pt",
            padding=True
        )
        
        # 分步处理图像以避免索引错误
        try:
            image_inputs = self.processor.image_processor(
                images, return_tensors="pt"
            )
            inputs['pixel_values'] = image_inputs['pixel_values']
        except Exception as e:
            print(f"Error processing images: {e}")
            print(f"Number of images: {len(images)}")
            # 如果图像处理失败，处理每个图像单独
            pixel_values_list = []
            for i, img in enumerate(images):
                try:
                    img_inputs = self.processor.image_processor([img], return_tensors="pt")
                    pixel_values = img_inputs['pixel_values']
                    print(f"Image {i} pixel_values shape: {pixel_values.shape}")
                    
                    # 检查形状并重塑为正确的4维张量
                    if len(pixel_values.shape) == 2:
                        # 如果是2维，需要重塑为合适的4维张量
                        # 假设这是已经flatten的图像数据，尝试重塑
                        # 这里可能需要根据实际的图像处理器输出调整
                        print(f"Warning: 2D pixel_values detected, shape: {pixel_values.shape}")
                        # 暂时跳过这个图像或使用默认处理
                        continue
                    else:
                        pixel_values_list.append(pixel_values)
                        
                except Exception as img_e:
                    print(f"Error processing individual image {i}: {img_e}")
                    continue
                    
            if pixel_values_list:
                try:
                    inputs['pixel_values'] = torch.cat(pixel_values_list, dim=0)
                except Exception as cat_e:
                    print(f"Error concatenating pixel_values: {cat_e}")
                    # 如果合并失败，使用第一个图像的值
                    inputs['pixel_values'] = pixel_values_list[0]
            else:
                print("No valid pixel_values found, returning zero embeddings")
                return torch.zeros(len(doc_indices), 4096)
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states[-1]
            doc_embeddings = hidden_states.mean(dim=1)  # [batch_size, hidden_size]
        
        return doc_embeddings
    
    def calculate_query_metrics(self, ranked_docs, relevant_docs):
        """计算单个查询的指标"""
        metrics = {}
        
        # 计算各种k值的指标
        for k in [1, 3, 5, 10]:
            hits_at_k = [doc in relevant_docs for doc in ranked_docs[:k]]
            
            # MRR@k
            if any(hits_at_k):
                first_hit_rank = hits_at_k.index(True) + 1
                metrics[f'MRR@{k}'] = 1.0 / first_hit_rank
            else:
                metrics[f'MRR@{k}'] = 0.0
            
            # Recall@k
            metrics[f'Recall@{k}'] = sum(hits_at_k) / len(relevant_docs)
            
            # Precision@k
            metrics[f'Precision@{k}'] = sum(hits_at_k) / k
        
        return metrics
    
    def save_results(self, dual_encoder_results, output_dir):
        """保存测试结果"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存指标
        results_summary = {
            'dual_encoder': dual_encoder_results[0]
        }
        
        with open(os.path.join(output_dir, 'performance_metrics.json'), 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")
        print("Performance Summary:")
        print("Dual-Encoder:", dual_encoder_results[0])

def main():
    parser = argparse.ArgumentParser(description='Test VLM performance on Qilin dataset')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to VLM model')
    parser.add_argument('--sample_num', type=int, default=100,
                       help='Number of test samples')
    parser.add_argument('--output_dir', type=str, default='vlm_test_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # 创建测试器
    tester = VLMPerformanceTester(args.model_path, args.device)
    
    # 测试交叉编码器性能
    # cross_encoder_results = tester.test_cross_encoder_performance(args.sample_num)
     # 测试双向编码器性能
    # dual_encoder_results = tester.test_dual_encoder_performance(args.sample_num)
    # 测试双向编码器性能 - 使用快速测试
    dual_encoder_results = tester.test_dual_encoder_quick(sample_num=args.sample_num, doc_limit=10)
    
    # 保存结果
    tester.save_results(dual_encoder_results, args.output_dir)

if __name__ == "__main__":
    main()
