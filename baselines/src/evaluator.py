import torch
import numpy as np
import os
from tqdm import tqdm
import pandas as pd
from collections import defaultdict
from utils import *
import editdistance
from rank_bm25 import BM25Okapi
from typing import List, Tuple
import jieba 
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity

def load_csv(file_path):
    """
    Load CSV file and return a mapping from qid to pids
    """
    df = pd.read_csv(file_path)
    qid_to_pids = defaultdict(list)
    for _, row in df.iterrows():
        qid_to_pids[int(row["qid"])].append(int(row["pid"]))
    return qid_to_pids

def calculate_metrics(sorted_results, ground_truth, k_list):
    """
    Calculate MRR@k, MAP@k, Recall@k, Precision@k for multiple k values

    Args:
        sorted_results (dict): Ranking results, qid -> [pid1, pid2, ...] (in ranked order)
        ground_truth (dict): Ground truth data, qid -> {pid1, pid2, ...} (set of positive samples)
        k_list (list): List of k values to calculate metrics for, e.g. [1, 3, 5, 10]

    Returns:
        dict: Average metrics, including MRR@k, MAP@k, Recall@k, Precision@k for each k value
    """
    max_k = max(k_list)  # Get maximum k value
    metrics = {k: {"mrr": 0.0, "map_sum": 0.0, "recall": 0.0, "precision": 0.0} for k in k_list}
    num_queries = len(ground_truth)
    valid_queries = 0

    for qid, relevant_pids in ground_truth.items():
        if qid not in sorted_results:
            num_queries -= 1  # Skip if qid not in ranking results
            continue
        
        valid_queries += 1
        retrieved_pids = sorted_results[qid][:max_k]
        hits = [pid in relevant_pids for pid in retrieved_pids]

        # Calculate metrics for each k value
        for k in k_list:
            hits_at_k = hits[:k]
            
            # Calculate MRR@k
            if any(hits_at_k):
                first_hit_rank = hits_at_k.index(True) + 1  # rank starts from 1
                metrics[k]["mrr"] += 1 / first_hit_rank

            # Calculate MAP@k
            avg_precision = 0.0
            num_correct = 0
            for i, is_relevant in enumerate(hits_at_k):
                if is_relevant:
                    num_correct += 1
                    precision_at_i = num_correct / (i + 1)
                    avg_precision += precision_at_i
            if num_correct > 0:  # Avoid division by zero
                avg_precision /= min(len(relevant_pids), k)  # Use min(|rel|, k) as denominator
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

class BM25Evaluator:
    def __init__(self, test_data, notes, qrels_data_path, result_key, **kwargs):
        self.qrels = load_csv(qrels_data_path)
        self.top_k = kwargs.get("top_k", 100)
        self.test_data = test_data
        self.notes = notes
        self.result_key = result_key

    def evaluate(self):
        """
        Evaluate retrieval results using the BM25 model.
        """
        metrics = None
        sorted_results = {}
        
        total_queries = len(self.test_data)
        pbar = tqdm(
            total=total_queries,
            desc="Evaluating BM25",
            unit="query",
            ncols=100,
            leave=True
        )

        for qid, sample in enumerate(self.test_data):
            pbar.set_description(f"Processing query {qid+1}/{total_queries}")
            
            search_results = sample[self.result_key]
            search_results = list(set(search_results))
            notes = [self.notes[note_idx] for note_idx in search_results]
            pid2idx = {pid: note_idx for pid, note_idx in enumerate(search_results)}

            bm25 = BM25Okapi(notes)
            query = sample["query"]

            scores = bm25.get_scores(query)
            sorted_pids = np.argsort(scores)[::-1][:self.top_k]
            sorted_pidxs = [pid2idx[pid] for pid in sorted_pids]
            sorted_results[qid] = sorted_pidxs

            pbar.update(1)
            
            pbar.set_postfix({
                'Current_Query_Length': len(query),
                'Results_Found': len(sorted_pidxs)
            })

        pbar.close()

        metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
        print("\nEvaluation Metrics:")
        print(metrics)
        
        return metrics

    
class DenseRetrievalEvaluator:
    def __init__(self, accelerator, model, passage_dataloader, question_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.passage_dataloader = passage_dataloader
        self.question_dataloader = question_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        self.top_k = kwargs.get("top_k", 100)
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def encode_dataloader(self, dataloader, output_file_prefix):
        """
        Encode data from DataLoader using multiple GPUs and save to corresponding files.

        Args:
            dataloader (DataLoader): Data loader.
            output_file_prefix (str): Output file name prefix.
        """
        local_rank = self.accelerator.process_index  # 当前设备的编号
        output_file = os.path.join(self.output_dir, f"{output_file_prefix}_gpu_{local_rank}.npy")
        all_hidden_states = []
        self.model.model.eval()

        with torch.no_grad():
            for batch in tqdm(dataloader):
                tokenized_inputs = batch["notes_tokenized"] if "notes_tokenized" in batch else batch["queries_tokenized"]
                tokenized_inputs = {key: val.to(self.accelerator.device) for key, val in tokenized_inputs.items()}
                hidden_states = self.model.forward(**tokenized_inputs)

                all_hidden_states.append(hidden_states.detach().cpu().numpy())

        all_hidden_states = np.concatenate(all_hidden_states, axis=0)
        np.save(output_file, all_hidden_states)
        print(f"Saved encoded data to {output_file}")

    def encode(self):
        """
        Encode questions and passages separately and save the results.
        """

        self.encode_dataloader(self.passage_dataloader, "passage")
        self.accelerator.wait_for_everyone()
        self.encode_dataloader( self.question_dataloader, "question")
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            all_hidden_states = []
            for i in range(self.accelerator.num_processes):
                output_file = os.path.join(self.output_dir, f"question_gpu_{i}.npy")
                all_hidden_states.append(np.load(output_file))
                os.remove(output_file)
            all_hidden_states = np.concatenate(all_hidden_states, axis=0)
            np.save(os.path.join(self.output_dir, "question.npy"), all_hidden_states)

    def retrieval(self):
        """
        Retrieve encoded data from files.
        """
        local_rank = self.accelerator.process_index
        passage_data = np.load(os.path.join(self.output_dir, f"passage_gpu_{local_rank}.npy"))
        question_data = np.load(os.path.join(self.output_dir, f"question.npy"))
        shift = 0
        for i in range(local_rank):
            shift += np.load(os.path.join(self.output_dir, f"passage_gpu_{i}.npy")).shape[0]
        passage_data = torch.tensor(passage_data).to(self.accelerator.device)
        batch_size = 128
        with open(os.path.join(self.output_dir, f"retrieval_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,similarity\n")
            for i in range(0, question_data.shape[0], batch_size):
                query = question_data[i:i+batch_size]
                query = torch.tensor(query).to(self.accelerator.device)
                sim = torch.matmul(query, passage_data.T)
                topk = torch.topk(sim, k=self.top_k, dim=1)
                for j in range(query.shape[0]):
                    for k in range(self.top_k):
                        f.write(f"{i+j},{topk.indices[j][k]+shift},{topk.values[j][k]}\n")
        # merge
        self.accelerator.wait_for_everyone()
        if self.accelerator.is_main_process:
            all_results = []
            for i in range(self.accelerator.num_processes):
                with open(os.path.join(self.output_dir, f"retrieval_results_gpu_{i}.csv"), "r") as f_gpu:
                    next(f_gpu)
                    for line in f_gpu:
                        qid, pid, sim = line.strip().split(',')
                        all_results.append((int(qid), int(pid), float(sim)))
                os.remove(os.path.join(self.output_dir, f"retrieval_results_gpu_{i}.csv"))
            
            results_dict = {}
            for qid, pid, sim in all_results:
                if qid not in results_dict:
                    results_dict[qid] = []
                results_dict[qid].append((pid, sim))
            
            with open(os.path.join(self.output_dir, "retrieval_results.csv"), "w") as f:
                f.write("qid,pid,similarity\n")
                for qid in sorted(results_dict.keys()):
                    results = sorted(results_dict[qid], key=lambda x: x[1], reverse=True)[:self.top_k]
                    for pid, sim in results:
                        f.write(f"{qid},{pid},{sim}\n")



    def evaluate(self):
        """
        Evaluate retrieval results.
        """
        self.encode()
        self.accelerator.wait_for_everyone()
        self.retrieval()
        self.accelerator.wait_for_everyone()
        metrics = None
        if self.accelerator.is_main_process:
            sorted_results = load_csv(os.path.join(self.output_dir, "retrieval_results.csv"))
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            print(metrics)
        return metrics


class DenseRetrievalRerankingEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        """
        Evaluate DR reranking performance.
        """
        local_rank = self.accelerator.process_index
        self.model.model.eval()
        
        predictions = defaultdict(list)
        
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
        
        self.accelerator.wait_for_everyone()
        
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        
        next_qid = shift
        search_idx_to_qid = {}
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                query_inputs = {key: val.to(self.accelerator.device) for key, val in batch['query_inputs'].items()}
                doc_inputs = {key: val.to(self.accelerator.device) for key, val in batch['doc_inputs'].items()}
                note_idxs = batch['note_idxs']
                search_idxs = batch['search_idxs']
                
                query_embeddings = self.model.forward(**query_inputs)
                doc_embeddings = self.model.forward(**doc_inputs)

                scores = torch.sum(query_embeddings * doc_embeddings, dim=1)
                
                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    if search_idx not in search_idx_to_qid:
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    predictions[qid].append((note_idx, scores[i].item()))
        
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        if self.accelerator.is_main_process:
            # Merge results from all GPUs
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f) 
                    for line in f:
                        qid, pid, score = line.strip().split(',')
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
                os.remove(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"))
            
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]
            
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            print(metrics)
        
        return metrics

class CrossEncoderEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        """
        Evaluate CrossEncoder model performance.
        """
        local_rank = self.accelerator.process_index
        self.model.eval()
        
        predictions = defaultdict(list)
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
        self.accelerator.wait_for_everyone()
        # Read and accumulate the sample counts from previous processes as shift
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        next_qid = shift
        search_idx_to_qid = {}
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                inputs = {key: val.to(self.accelerator.device) for key, val in batch['inputs'].items()}
                note_idxs = batch['note_idxs']
                search_idxs = batch['search_idxs']
                
                outputs = self.model(**inputs)
                scores = outputs.squeeze(-1)
                        
                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    if search_idx not in search_idx_to_qid:
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    predictions[qid].append((note_idx, scores[i].item()))
        # Write results to file
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        if self.accelerator.is_main_process:
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f)  
                    for line in f:
                        qid, pid, score = line.strip().split(',') 
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
            # Write final results
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            # Group and sort by query ID
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            # Sort results for each query by score
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]
            
            # Calculate evaluation metrics
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            print(metrics)
        
        return metrics

class DCNEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        """
        Evaluate SearchDCN model performance.
        """
        if self.accelerator is None:
            self.model.eval()
            predictions = defaultdict(list)
            search_idx_to_qid = {}
            next_qid = 0
            
            with torch.no_grad():
                for batch in tqdm(self.test_dataloader):
                    query_features = {k: v.to(next(self.model.parameters()).device) for k, v in batch[0].items()}
                    user_features = {k: v.to(next(self.model.parameters()).device) for k, v in batch[1].items()}
                    note_features = {k: v.to(next(self.model.parameters()).device) for k, v in batch[2].items()}
                    note_idxs = batch[3]['note_idxs']
                    search_idxs = batch[3]['search_idxs']
                    
                    scores = self.model(query_features, user_features, note_features)
                    scores = scores.squeeze(-1)
                    
                    for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                        if search_idx not in search_idx_to_qid:
                            search_idx_to_qid[search_idx] = next_qid
                            next_qid += 1
                        qid = search_idx_to_qid[search_idx]
                        predictions[qid].append((note_idx, scores[i].item()))
            
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, preds in predictions.items():
                    for pid, score in preds:
                        f.write(f"{qid},{pid},{score}\n")
            
            sorted_results = {}
            for qid, preds in predictions.items():
                sorted_results[qid] = [pid for pid, _ in sorted(preds, key=lambda x: x[1], reverse=True)]
            
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            print(metrics)
            return metrics
            
        local_rank = self.accelerator.process_index
        self.model.eval()
        
        predictions = defaultdict(list)
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
        self.accelerator.wait_for_everyone()
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        next_qid = shift
        search_idx_to_qid = {}
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader):
                query_features = {k: v.to(self.accelerator.device) for k, v in batch[0].items()}
                user_features = {k: v.to(self.accelerator.device) for k, v in batch[1].items()}
                note_features = {k: v.to(self.accelerator.device) for k, v in batch[2].items()}
                note_idxs = batch[3]['note_idxs']
                search_idxs = batch[3]['search_idxs']
                
                scores = self.model(query_features, user_features, note_features)
                scores = scores.squeeze(-1).reshape(-1)
                        
                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    if search_idx not in search_idx_to_qid:
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    predictions[qid].append((note_idx, scores[i].item()))
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        if self.accelerator.is_main_process:
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f)  
                    for line in f:
                        qid, pid, score = line.strip().split(',') 
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]
            
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            print(metrics)
        
        return metrics


class VLMCrossEncoderEvaluator:
    def __init__(self, accelerator, model, test_dataloader, **kwargs):
        self.accelerator = accelerator
        self.model = model
        self.test_dataloader = test_dataloader
        self.output_dir = kwargs.get("output_dir", "output")
        qrels_data_path = kwargs.get("qrels_data_path", "data/qrels.csv")
        self.qrels = load_csv(qrels_data_path)
        os.makedirs(self.output_dir, exist_ok=True)

    def evaluate(self):
        """
        Evaluate VLM CrossEncoder model performance.
        """
        local_rank = self.accelerator.process_index
        self.accelerator.wait_for_everyone()
        self.model.eval()
        
        # Store prediction results for each query
        predictions = defaultdict(list)
        
        # Record current process's sample count
        local_samples = len(self.test_dataloader.dataset)
        with open(os.path.join(self.output_dir, f"samples_count_gpu_{local_rank}.txt"), "w") as f:
            f.write(str(local_samples))
            
        self.accelerator.wait_for_everyone()
        
        # Calculate offset
        shift = 0
        for i in range(local_rank):
            with open(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"), "r") as f:
                shift += int(f.read().strip())
        
        next_qid = shift
        search_idx_to_qid = {}
        
        with torch.no_grad():
            for batch in tqdm(self.test_dataloader, desc=f"Evaluating on GPU {local_rank}"):
                inputs = {
                    key: val.to(self.accelerator.device)
                    for key, val in batch['inputs'].items() 
                }
                note_idxs = batch['note_idxs']
                search_idxs = batch['search_idxs']
                
                # Set mini-batch size
                mini_batch_size = 20
                batch_size = len(note_idxs)
                scores_list = []
                images_per_text = inputs['pixel_values'].size(0) // batch_size

                # Process by mini-batch
                for i in range(0, batch_size, mini_batch_size):
                    mini_batch_inputs = {
                        k: v[i:i+mini_batch_size] for k,v in inputs.items() if k != 'pixel_values'
                    }
                    mini_batch_inputs['pixel_values'] = inputs['pixel_values'][i*images_per_text:(i+mini_batch_size)*images_per_text]
                    mini_outputs = self.model(**mini_batch_inputs)
                    mini_scores = mini_outputs.squeeze(-1)
                    scores_list.append(mini_scores)
                
                # Merge results from all mini-batches
                scores = torch.cat(scores_list, dim=0).reshape(-1)

                
                for i, (note_idx, search_idx) in enumerate(zip(note_idxs, search_idxs)):
                    # Assign qid for new search_idx
                    if search_idx not in search_idx_to_qid:
                        search_idx_to_qid[search_idx] = next_qid
                        next_qid += 1
                    qid = search_idx_to_qid[search_idx]
                    predictions[qid].append((note_idx, scores[i].item()))
                
        
        # Write results to file
        with open(os.path.join(self.output_dir, f"rerank_results_gpu_{local_rank}.csv"), "w") as f:
            f.write("qid,pid,score\n")
            for qid, preds in predictions.items():
                for pid, score in preds:
                    f.write(f"{qid},{pid},{score}\n")
        
        self.accelerator.wait_for_everyone()
        
        metrics = None
        if self.accelerator.is_main_process:
            # Merge results from all GPUs
            all_results = []
            for i in range(self.accelerator.num_processes):
                result_file = os.path.join(self.output_dir, f"rerank_results_gpu_{i}.csv")
                with open(result_file, "r") as f:
                    next(f) 
                    for line in f:
                        qid, pid, score = line.strip().split(',')
                        all_results.append((int(qid), int(pid), float(score)))
                os.remove(result_file)
            
            # Clean up temporary files
            for i in range(self.accelerator.num_processes):
                os.remove(os.path.join(self.output_dir, f"samples_count_gpu_{i}.txt"))
            
            # Write final results
            with open(os.path.join(self.output_dir, "rerank_results.csv"), "w") as f:
                f.write("qid,pid,score\n")
                for qid, pid, score in all_results:
                    f.write(f"{qid},{pid},{score}\n")
            
            # Group and sort by query ID
            sorted_results = {}
            for qid, pid, score in all_results:
                if qid not in sorted_results:
                    sorted_results[qid] = []
                sorted_results[qid].append((pid, score))
            
            # Sort results for each query by score
            for qid in sorted_results:
                sorted_results[qid] = [pid for pid, _ in sorted(sorted_results[qid], key=lambda x: x[1], reverse=True)]
            
            # Calculate evaluation metrics
            metrics = calculate_metrics(sorted_results, self.qrels, [10, 100])
            
            # Print detailed evaluation results
            print("\nEvaluation Results:")
            print("=" * 50)
            for k, v in metrics.items():
                print(f"{k}: {v:.4f}")
            print("=" * 50)
        
        return metrics

