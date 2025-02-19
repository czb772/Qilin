import os
import torch
import random
import numpy as np
import pandas as pd
from PIL import Image
from utils import *
from transformers import AutoTokenizer, AutoProcessor
from datasets import load_from_disk, load_dataset
from torch.utils.data import DataLoader
from registry import register_class
@register_class
class DenseRetrievalTrainingDataProcessor:
    def __init__(self, **kwargs):
        """
        Data processor class for loading and processing data.

        Args:
            data_path (str): Dataset path (supports load_from_disk format).
            tokenizer_name (str): Name of the pretrained tokenizer.
            batch_size (int): Batch size.
            max_length (int): Maximum length for tokenizer.
        """
        data_path = kwargs.get('dataset_name_or_path')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        max_length = kwargs.get('max_length')
        negative_samples = kwargs.get('negative_samples')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.N = negative_samples
        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'

    def load_data(self):
        """
        Load dataset from disk.
        Returns:
            datasets.Dataset: The loaded dataset.
        """
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        self.dataset = dataset[self.train_data_key]

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def collate_fn(self, batch):
        """
        Batch data processing function.
        Args:
            batch (list): Batch data.
        Returns:
            Tuple[Dict, List]: Tokenized batch inputs and targets.
        """
        queries = [item["query"] for item in batch]
        notes = []
        note_idxs = []

        for item in batch:
            # Randomly select an index from positives
            impression_result_details = item[self.negative_pool]
            positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            positive_idx = random.randint(0, len(positives) - 1)
            note_idx = positives[positive_idx]
            positive_note = self.get_note_content(note_idx)
            notes.append(positive_note)
            note_idxs.append(note_idx)
            
            # Randomly select N indices from negatives
            negatives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] != 0]
            if len(negatives) < self.N:
                # Randomly select from all notes
                negatives = negatives + random.sample(range(len(self.corpus)), k=self.N-len(negatives))
            else:
                negatives = random.sample(negatives, k=self.N)
            notes.extend([self.get_note_content(note_idx) for note_idx in negatives])
            note_idxs.extend(negatives)

        queries_tokenized = self.tokenizer(
            queries, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt")
        
        notes_tokenized = self.tokenizer(
            notes, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt"
        )
        queries_and_notes = queries + notes
        merged_tokenized = self.tokenizer(
            queries_and_notes, 
            truncation=True, 
            padding=True, 
            max_length=self.max_length, 
            return_tensors="pt")
        
        # 返回分词后的问题和段落
        return {'queries_tokenized':queries_tokenized, 'notes_tokenized':notes_tokenized, 'merged_tokenized':merged_tokenized}

    def get_dataloader(self):
        """
        Get DataLoader.
        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )
    
@register_class
class DenseRetrievalRerankingTestDataProcessor:
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.max_length = kwargs.get('max_length')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        self.dataset = self.load_data()

    def load_data(self):
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        data = dataset[self.test_data_key]
        data = data.select(range(min(self.sample_num, len(data))))
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def collate_fn(self, batch):
        queries = []
        notes = []
        note_idxs = []
        search_idxs = []

        for item in batch:
            query = item["query"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            # Use specified recall results
            candidates = item[self.results_key]
            # Sort candidate documents by score in descending order
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            # If rerank_depth is set, only take the top k candidates
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            
            for candidate in candidates:
                note_idx = int(candidate[0])  # First element is note_idx
                note_content = self.get_note_content(note_idx)
                
                queries.append(query)
                notes.append(note_content)
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)

        # Process queries and documents separately for tokenization
        query_inputs = self.tokenizer(
            queries,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        doc_inputs = self.tokenizer(
            notes,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "query_inputs": query_inputs,
            "doc_inputs": doc_inputs,
            "note_idxs": note_idxs,
            "search_idxs": search_idxs
        }

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )

@register_class
class NoteDataProcessor:
    def __init__(self, local_rank, num_processes, **kwargs):
        """
        Note data processor class for loading and processing passage data.

        Args:
            data_path (str): Dataset path (supports load_from_disk format).
            tokenizer_name (str): Name of the pretrained tokenizer.
            batch_size (int): Batch size.
            max_length (int): Maximum length for tokenizer.
        """
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('batch_size')
        self.max_length = kwargs.get('max_length')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.dataset = self.load_data()

    def load_data(self):
        """
        Load dataset from disk.
        Returns:
            datasets.Dataset: The loaded dataset.
        """
        dataset = load_from_disk(self.data_path)['notes'].shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        self.corpus = load_from_disk(self.data_path)['notes']
        return dataset
    
    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def collate_fn(self, batch):
        """
        Batch data processing function.
        Args:
            batch (list): Batch data.
        Returns:
            Dict[str, Dict]: Tokenized passages.
        """
        note_idxs = [item["note_idx"] for item in batch]
        notes = [self.get_note_content(note_idx) for note_idx in note_idxs]


        # Tokenize passages
        notes_tokenized = self.tokenizer(
            notes,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {"notes_tokenized": notes_tokenized, "note_idxs":note_idxs, "notes":notes}

    def get_dataloader(self):
        """
        Get DataLoader.
        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
@register_class
class QueryDataProcessor:
    def __init__(self, local_rank, num_processes, **kwargs):
        """
        Query data processor class for loading and processing question data.

        Args:
            data_path (str): Dataset path (supports load_from_disk format).
            tokenizer_name (str): Name of the pretrained tokenizer.
            batch_size (int): Batch size.
            max_length (int): Maximum length for tokenizer.
        """
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('batch_size')
        self.max_length = kwargs.get('max_length')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.dataset = self.load_data()

    def load_data(self):
        """
        Load dataset from disk.
        Returns:
            datasets.Dataset: The loaded dataset.
        """
        dataset = load_from_disk(self.data_path)['search_test'].shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return dataset

    def collate_fn(self, batch):
        """
        批量数据处理函数。
        Args:
            batch (list): 批量数据。
        Returns:
            Dict[str, Dict]: 分词后的问题和附加信息。
        """
        queries = [item["query"] for item in batch]

        # 分词问题
        queries_tokenized = self.tokenizer(
            queries,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {"queries": queries, "queries_tokenized": queries_tokenized}

    def get_dataloader(self):
        """
        Get DataLoader.
        Returns:
            DataLoader: PyTorch DataLoader object.
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    

@register_class
class DCNTrainingDataProcessor:
    def __init__(self, **kwargs):
        """
        Data processor class for DCN search model, used for loading and processing training data.

        Args:
            dataset_name_or_path (str): Dataset path
            batch_size (int): Batch size
            negative_samples (int): Number of negative samples per query
        """
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('batch_size')
        self.negative_samples = kwargs.get('negative_samples', 3)
        self.max_length = kwargs.get('max_length', 512)
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        self.tokenizer.truncation_side = 'right'
        self.tokenizer.padding_side = 'right'
        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret
    
    def load_data(self):
        """
        Load dataset from disk
        """
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        self.dataset = dataset[self.train_data_key]
        self.user_features = dataset['user_feat']

    def get_note_dense_features(self, note_idx):
        """
        Get dense features of the note
        """
        note = self.corpus[note_idx]
        note_dense_feature_names = ['video_duration', 'video_height', 'video_width', 'image_num', 
                                    'content_length', 'commercial_flag', 'imp_num', 'imp_rec_num', 
                                    'imp_search_num', 'click_num', 'click_rec_num', 'click_search_num', 
                                    'like_num', 'collect_num', 'comment_num', 'share_num', 'screenshot_num',
                                    'hide_num', 'rec_like_num', 'rec_collect_num', 'rec_comment_num', 
                                    'rec_share_num', 'rec_follow_num', 'search_like_num', 
                                    'search_collect_num', 'search_comment_num', 'search_share_num', 
                                    'search_follow_num', 'accum_like_num', 'accum_collect_num', 
                                    'accum_comment_num', 'view_time', 'rec_view_time', 'search_view_time', 
                                    'valid_view_times', 'full_view_times']
        features = [note[feature_name] for feature_name in note_dense_feature_names]
        features = [0.0 if pd.isna(x) else x for x in features]
        return torch.tensor(features, dtype=torch.float32)

    def get_note_sparse_features(self, note_idx):
        """
        Get sparse features of the note
        """
        note = self.corpus[note_idx]
        return {
            'note_type': torch.tensor(note['note_type'], dtype=torch.long),
            'taxonomy1_id': torch.tensor(hash(note['taxonomy1_id']) % 43, dtype=torch.long),
            'taxonomy2_id': torch.tensor(hash(note['taxonomy2_id']) % 311, dtype=torch.long),
            'taxonomy3_id': torch.tensor(hash(note['taxonomy3_id']) % 548, dtype=torch.long),
            'note_idx': torch.tensor(note_idx, dtype=torch.long)
        }

    def get_user_dense_features(self, user_idx):
        """
        Get user dense feature
        """
        user = self.user_features[user_idx]
        dense_features = [user[f'dense_feat{i}'] for i in range(1, 41)]
        dense_features.extend([user['fans_num'], user['follows_num']])
        dense_features = [0.0 if pd.isna(x) else x for x in dense_features]
        return torch.tensor(dense_features, dtype=torch.float32)

    def get_user_sparse_features(self, user_idx):
        """
        Get user sparse feature
        """
        user = self.user_features[user_idx]
        gender_map = {'male': 0, 'female': 1, 'unknown': 2}
        platform_map = {'iOS': 0, 'Android': 1, 'Harmony': 2, 'unknown': 3}
        age_map = {
            '1-12': 0, '13-15': 1, '16-18': 2, '19-22': 3, '23-25': 4,
            '26-30': 5, '31-35': 6, '36-40': 7, '40+':8, 'unknown': 9
        }
        return {
            'gender': torch.tensor(gender_map.get(user['gender'], 2), dtype=torch.long),
            'platform': torch.tensor(platform_map.get(user['platform'], 3), dtype=torch.long),
            'age': torch.tensor(age_map.get(user['age'], 9), dtype=torch.long),
            'user_idx': torch.tensor(user_idx, dtype=torch.long),
            'location': torch.tensor(hash(user['location']) % 1096 if user['location'] else 0, dtype=torch.long)
        }

    def collate_fn(self, batch):
        """
        Batch data processing function
        """
        query_features = {'question_embedding': [], 'query_from_type': []}
        user_features = {'dense': [], 'recent_clicked_note_idxs': []}
        note_features = {'note_embedding': [], 'dense': []}
        labels = []

        for item in batch:
            # process positive and negative samples
            impression_result_details = item[self.negative_pool]
            positives = [r['note_idx'] for r in impression_result_details if r['click'] == 1]
            negatives = [r['note_idx'] for r in impression_result_details if r['click'] == 0]

            # process positive sample
            if positives:
                pos_idx = random.choice(positives)
                query_features['query_from_type'].append(torch.tensor(item.get('query_from_type', 15), dtype=torch.long))
                query_features['question_embedding'].append(item['question_embedding'])

                user_idx = item['user_idx']
                user_dense = self.get_user_dense_features(user_idx)
                user_sparse = self.get_user_sparse_features(user_idx)
                user_features['dense'].append(user_dense)
                for k, v in user_sparse.items():
                    if k not in user_features:
                        user_features[k] = []
                    user_features[k].append(v)
                
                recent_notes = item['recent_clicked_note_idxs'][:10]  # 只取前10个
                if len(recent_notes) < 10:
                    recent_notes = recent_notes + [1983938] * (10 - len(recent_notes))  # 补1983938作为填充
                user_features['recent_clicked_note_idxs'].append(torch.tensor(recent_notes))

                note_dense = self.get_note_dense_features(pos_idx)
                note_sparse = self.get_note_sparse_features(pos_idx)
                note_features['dense'].append(note_dense)
                note_features['note_embedding'].append(self.corpus[pos_idx]['note_embedding'])
                for k, v in note_sparse.items():
                    if k not in note_features:
                        note_features[k] = []
                    note_features[k].append(v)
                labels.append(1)

                if len(negatives) < self.negative_samples:
                    additional = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                    negatives.extend(additional)
                selected_negs = random.sample(negatives, k=self.negative_samples)
                
                for neg_idx in selected_negs:
                    query_features['query_from_type'].append(torch.tensor(item.get('query_from_type', 15), dtype=torch.long))
                    query_features['question_embedding'].append(item['question_embedding'])

                    user_features['dense'].append(user_dense)
                    for k, v in user_sparse.items():
                        user_features[k].append(v)
                    user_features['recent_clicked_note_idxs'].append(torch.tensor(recent_notes))

                    note_dense = self.get_note_dense_features(neg_idx)
                    note_sparse = self.get_note_sparse_features(neg_idx)
                    note_features['dense'].append(note_dense)
                    note_features['note_embedding'].append(self.corpus[neg_idx]['note_embedding'])
                    for k, v in note_sparse.items():
                        note_features[k].append(v)
                    labels.append(0)

        query_features['query_from_type'] = torch.stack([torch.tensor(x) for x in query_features['query_from_type']])
        query_features['question_embedding'] = torch.stack([torch.tensor(x) for x in query_features['question_embedding']])
        user_features['dense'] = torch.stack(user_features['dense'])
        user_features['recent_clicked_note_idxs'] = torch.stack(user_features['recent_clicked_note_idxs'])
        for k in ['gender', 'platform', 'age', 'user_idx', 'location']:
            user_features[k] = torch.stack(user_features[k])
        note_features['dense'] = torch.stack(note_features['dense'])
        note_features['note_embedding'] = torch.stack([torch.tensor(x) for x in note_features['note_embedding']])
        for k in ['note_type', 'taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id', 'note_idx']:
            note_features[k] = torch.stack(note_features[k])
        labels = torch.tensor(labels, dtype=torch.float32)
        
        def check_and_fix_nan(tensor):
            if torch.isnan(tensor).any():
                return torch.nan_to_num(tensor, nan=0.0)
            return tensor

        query_features['query_from_type'] = check_and_fix_nan(query_features['query_from_type'])

        user_features['dense'] = check_and_fix_nan(user_features['dense'])
        user_features['recent_clicked_note_idxs'] = check_and_fix_nan(user_features['recent_clicked_note_idxs'])
        for k in ['gender', 'platform', 'age', 'user_idx', 'location']:
            user_features[k] = check_and_fix_nan(user_features[k])

        note_features['dense'] = check_and_fix_nan(note_features['dense'])
        for k in ['note_type', 'taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id', 'note_idx']:
            note_features[k] = check_and_fix_nan(note_features[k])

        labels = check_and_fix_nan(labels)
        
        return query_features, user_features, note_features, labels

    def get_dataloader(self):
        """
        Get DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn
        )

@register_class
class CrossEncoderTrainingDataProcessor:
    def __init__(self, **kwargs):
        data_path = kwargs.get('dataset_name_or_path')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        max_length = kwargs.get('max_length')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.negative_samples = kwargs.get('negative_samples', 3) 
        self.tokenizer_name = tokenizer_name
        self.data_path = data_path
        self.batch_size = batch_size
        self.max_length = max_length
        self.load_data()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'

    def load_data(self):
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        self.dataset = dataset[self.train_data_key]

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def collate_fn(self, batch):
        queries = []
        notes = []
        labels = []

        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            
            positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            assert len(positives) > 0, 'No positive samples found for query: ' + query
            positive_idx = random.choice(positives)
            note_content = self.get_note_content(positive_idx)
            queries.append(query)
            notes.append(note_content)
            labels.append(1)
        
            negatives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 0]
            if len(negatives) < self.negative_samples:
                additional_samples = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                negatives.extend(additional_samples)
            else:
                negatives = random.sample(negatives, k=self.negative_samples)
            
            for note_idx in negatives:
                note_content = self.get_note_content(note_idx)
                queries.append(query)
                notes.append(note_content)
                labels.append(0)

        query_note_pairs = [f"{q} [SEP] {n}" for q, n in zip(queries, notes)]
        
        inputs = self.tokenizer(
            query_note_pairs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = torch.tensor(labels, dtype=torch.float)
        
        return {"inputs": inputs, "labels": labels}

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

@register_class
class DCNTestDataProcessor(DCNTrainingDataProcessor):
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        """
        Data processor class for DCN search model, used for loading and processing test data.

        Args:
            local_rank (int): Local rank of current process
            num_processes (int): Total number of processes
            results_key (str): Key name for recall results
            rerank_depth (int): Reranking depth
            dataset_name_or_path (str): Dataset path
            eval_batch_size (int): Evaluation batch size
        """
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        kwargs['batch_size'] = kwargs.get('eval_batch_size')
        self.sample_num = kwargs.get('sample_num')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.local_rank = local_rank
        self.num_processes = num_processes
        super().__init__(**kwargs)

    def load_data(self):
        """
        Load dataset from disk
        """
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        self.user_features = dataset['user_feat']
        data = dataset[self.test_data_key]
        data = data.select(range(min(self.sample_num, len(data))))
        self.dataset = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)

    def collate_fn(self, batch):
        """
        Batch data processing function
        """
        query_features = {'question_embedding': [], 'query_from_type': []}
        user_features = {'dense': [], 'recent_clicked_note_idxs': []}
        note_features = {'note_embedding': [], 'dense': []}
        note_idxs = []
        search_idxs = []

        for item in batch:
            query = item["query"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            user_idx = item['user_idx']

            user_dense = self.get_user_dense_features(user_idx)
            user_sparse = self.get_user_sparse_features(user_idx)
            
            recent_notes = item['recent_clicked_note_idxs'][:10]  # 只取前10个
            if len(recent_notes) < 10:
                recent_notes = recent_notes + [1983938] * (10 - len(recent_notes))  # 补1983938
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            
            for candidate in candidates:
                if type(candidate) == int:
                    candidate = [candidate, 0.0]
                note_idx = int(candidate[0])
                note_dense = self.get_note_dense_features(note_idx)
                note_sparse = self.get_note_sparse_features(note_idx)
                note_features['dense'].append(note_dense)
                note_features['note_embedding'].append(self.corpus[note_idx]['note_embedding'])
                for k, v in note_sparse.items():
                    if k not in note_features:
                        note_features[k] = []
                    note_features[k].append(v)
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)

                query_features['question_embedding'].append(item['question_embedding'])
                query_features['query_from_type'].append(torch.tensor(item.get('query_from_type', 15), dtype=torch.long))
                user_features['dense'].append(user_dense)
                for k, v in user_sparse.items():
                    if k not in user_features:
                        user_features[k] = []
                    user_features[k].append(v)
                user_features['recent_clicked_note_idxs'].append(torch.tensor(recent_notes))

        query_features['query_from_type'] = torch.stack([x for x in query_features['query_from_type']])
        query_features['question_embedding'] = torch.stack([torch.tensor(x) for x in query_features['question_embedding']])
        user_features['dense'] = torch.stack(user_features['dense'])
        user_features['recent_clicked_note_idxs'] = torch.stack(user_features['recent_clicked_note_idxs'])
        for k in ['gender', 'platform', 'age', 'user_idx', 'location']:
            user_features[k] = torch.stack(user_features[k])
        note_features['dense'] = torch.stack(note_features['dense'])
        for k in ['note_type', 'taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id', 'note_idx']:
            note_features[k] = torch.stack(note_features[k])
        note_features['note_embedding'] = torch.stack([torch.tensor(x) for x in note_features['note_embedding']])

        def check_and_fix_nan(tensor):
            if torch.isnan(tensor).any():
                return torch.nan_to_num(tensor, nan=0.0)
            return tensor

        query_features['query_from_type'] = check_and_fix_nan(query_features['query_from_type'])

        user_features['dense'] = check_and_fix_nan(user_features['dense'])
        user_features['recent_clicked_note_idxs'] = check_and_fix_nan(user_features['recent_clicked_note_idxs'])
        for k in ['gender', 'platform', 'age', 'user_idx', 'location']:
            user_features[k] = check_and_fix_nan(user_features[k])

        note_features['dense'] = check_and_fix_nan(note_features['dense'])
        for k in ['note_type', 'taxonomy1_id', 'taxonomy2_id', 'taxonomy3_id', 'note_idx']:
            note_features[k] = check_and_fix_nan(note_features[k])

        return query_features, user_features, note_features, {'note_idxs': note_idxs, 'search_idxs': search_idxs}

    def get_dataloader(self):
        """
        Get DataLoader
        """
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn
        )

@register_class
class CrossEncoderTestDataProcessor:
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.max_length = kwargs.get('max_length')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        tokenizer_name = kwargs.get('tokenizer_name_or_path')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.truncation_side = 'left' 
        self.tokenizer.padding_side = 'left'
        if "bert" in tokenizer_name or "bge" in tokenizer_name:
            self.tokenizer.truncation_side = 'right'
            self.tokenizer.padding_side = 'right'
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        self.dataset = self.load_data()

    def load_data(self):
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        data = dataset[self.test_data_key]
        data = data.select(range(min(self.sample_num, len(data))))
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data

    def get_note_content(self, note_idx):
        ret = ''
        if self.use_title:
            ret += self.corpus[note_idx]['note_title']
        if self.use_content:
            ret += self.corpus[note_idx]['note_content']
        return ret

    def collate_fn(self, batch):
        queries = []
        notes = []
        note_idxs = []
        search_idxs = []

        for item in batch:
            query = item["query"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            
            for candidate in candidates:
                note_idx = int(candidate[0])  
                note_content = self.get_note_content(note_idx)
                
                queries.append(query)
                notes.append(note_content)
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)

        query_note_pairs = [f"{q} [SEP] {n}" for q, n in zip(queries, notes)]
        
        inputs = self.tokenizer(
            query_note_pairs,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {"inputs": inputs, "note_idxs": note_idxs, "search_idxs": search_idxs}

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
    
@register_class
class VLMCrossEncoderTrainingDataProcessor:
    def __init__(self, **kwargs):
        data_path = kwargs.get('dataset_name_or_path')
        processor_name = kwargs.get('tokenizer_name_or_path')
        batch_size = kwargs.get('batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.max_length = kwargs.get('max_length', 1024)
        self.negative_samples = kwargs.get('negative_samples', 3)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        
        self.processor_name = processor_name
        self.data_path = data_path
        self.batch_size = batch_size

        self.train_data_key = kwargs.get('train_data_key', 'search_train')
        self.negative_pool = kwargs.get('negative_pool', 'search_result_details_with_idx')
        self.load_data()
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels= 100 * 28 * 28,
            trust_remote_code=True
        )
        self.default_image = self._create_default_image()

    def _create_default_image(self):
        # create a default image with white color
        default_image = Image.new('RGB', (1024, 1024), color='white')
        return default_image
    
    def load_data(self):
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        self.dataset = dataset[self.train_data_key]

    def get_note_content(self, note_idx):
        note = self.corpus[note_idx]
        image = self.default_image
        image_path = note['image_path']
        if len(image_path):
            try:
                image_path = os.path.join('/mnt/ali-sh-1/usr/lihaitao/process_0106', image_path[0])
                image = Image.open(image_path)
                image = image.resize((1024, 1024))
                image_size = image.size
                if image_size[0]<=0 or image_size[1]<=0:
                    image = self.default_image
            except Exception as e:
                print(f"Warning: Failed to load image for note {note_idx}: {e}")
            
        return {
            'text': self._get_text_content(note),
            'image': image
        }
    
    def _get_text_content(self, note):
        ret = ''
        if self.use_title:
            ret += note['note_title']
        if self.use_content:
            ret += note['note_content']
        return ret

    def collate_fn(self, batch):
        queries = []
        images = []
        labels = []

        for item in batch:
            query = item["query"]
            impression_result_details = item[self.negative_pool]
            
            positives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 1]
            assert len(positives) > 0, 'No positive samples found for query: ' + query
            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            positive_idx = random.choice(positives)
            note_content = self.get_note_content(positive_idx)
            
            # Template of conversation
            conversation = [{
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                ]
            }]
            
            queries.append(conversation)
            images.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
            labels.append(1)
            
            negatives = [impression_result['note_idx'] for impression_result in impression_result_details if impression_result['click'] == 0]
            if len(negatives) < self.negative_samples:
                additional_samples = random.sample(range(len(self.corpus)), k=self.negative_samples-len(negatives))
                negatives.extend(additional_samples)
            else:
                negatives = random.sample(negatives, k=self.negative_samples)
            
            for note_idx in negatives:
                note_content = self.get_note_content(note_idx)
                
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                    ]
                }]
                
                queries.append(conversation)
                images.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                labels.append(0)

        text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in queries]
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        labels = torch.tensor(labels, dtype=torch.float)
        
        return {"inputs": inputs, "labels": labels}

    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
        )

@register_class
class VLMCrossEncoderTestDataProcessor(VLMCrossEncoderTrainingDataProcessor):
    def __init__(self, local_rank, num_processes, results_key, rerank_depth, **kwargs):
        self.data_path = kwargs.get('dataset_name_or_path')
        self.batch_size = kwargs.get('eval_batch_size')
        self.use_title = kwargs.get('use_title')
        self.use_content = kwargs.get('use_content')
        self.results_key = results_key
        self.rerank_depth = rerank_depth
        self.sample_num = kwargs.get('sample_num')
        self.max_length = kwargs.get('max_length', 1024)
        self.num_machines = kwargs.get('num_machines', 0)
        self.machine_rank = kwargs.get('machine_rank', 0)
        self.use_recent_clicked_note_images = kwargs.get('use_recent_clicked_note_images', False)
        
        processor_name = kwargs.get('tokenizer_name_or_path')
        self.processor = AutoProcessor.from_pretrained(
            processor_name,
            max_pixels=10 * 28 * 28,
            trust_remote_code=True
        )
        
        self.local_rank = local_rank
        self.num_processes = num_processes
        self.test_data_key = kwargs.get('test_data_key', 'search_test')
        self.dataset = self.load_data()
        self.default_image = self._create_default_image()

    def load_data(self):
        dataset = load_from_disk(self.data_path)
        self.corpus = dataset['notes']
        data = dataset[self.test_data_key]
        data = data.select(range(min(self.sample_num, len(data))))
        data = data.shard(num_shards=self.num_processes, index=self.local_rank, contiguous=True)
        return data
    
    def collate_fn(self, batch):
        queries = []
        images = []
        note_idxs = []
        search_idxs = []

        for item in batch:
            query = item["query"]
            search_idx = item['search_idx'] if 'search_idx' in item else item['request_idx']
            candidates = item[self.results_key]
            if type(candidates[0]) == int:
                candidates = [[x, 0.0] for x in candidates]
            candidates = sorted(candidates, key=lambda x: x[1], reverse=True)
            
            if self.rerank_depth is not None:
                candidates = candidates[:self.rerank_depth]
            if self.use_recent_clicked_note_images:
                recent_clicked_note_idxs = item.get('recent_clicked_note_idxs', [])[:10]
                recent_clicked_note_images = []
                for note_idx in recent_clicked_note_idxs:
                    note_content = self.get_note_content(note_idx)
                    recent_clicked_note_images.append(note_content['image'])
                if len(recent_clicked_note_images):
                    query_image = vertical_concat_images(recent_clicked_note_images)
                else:
                    query_image = self.default_image

            for candidate in candidates:
                note_idx = int(candidate[0])
                note_content = self.get_note_content(note_idx)
                
                conversation = [{
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": f"用户的问题是：{query}\n笔记内容是：{note_content['text']}\n请你判断笔记是否相关，如果图片不是空白，则也考虑图片内容。"}
                    ]
                }]
                
                queries.append(conversation)
                images.append(vertical_concat_images([query_image, note_content['image']]) if self.use_recent_clicked_note_images else note_content['image'])
                note_idxs.append(note_idx)
                search_idxs.append(search_idx)

        text_prompts = [self.processor.apply_chat_template(q, add_generation_prompt=True) for q in queries]
        inputs = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "inputs": inputs,
            "note_idxs": note_idxs,
            "search_idxs": search_idxs
        }
    
    def get_dataloader(self):
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=self.collate_fn,
        )
