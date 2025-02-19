from transformers import AutoModel, AutoTokenizer
from datasets import load_from_disk
from utils import mean_token_pool
from tqdm import tqdm
import numpy as np
import torch

model = AutoModel.from_pretrained("/mnt/ali-sh-1/usr/dongqian/qilinbaselines/output_search_dpr/2025-02-08-07-08-28/bert_checkpoints")
tokenizer = AutoTokenizer.from_pretrained("../model/bert-base-chinese/")    
model = model.cuda()
model.eval()
qilin = load_from_disk('../datasets/qilin_data_training_0211/')

recommendation_train = qilin['recommendation_train']
recommendation_test = qilin['recommendation_test']

recommendation_train_query = recommendation_train['query']
recommendation_test_query = recommendation_test['query']

batch_size = 32
# Encode recommendation_train_query to recommendation_train_query_embeddings.npy
recommendation_train_query_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(recommendation_train_query), batch_size)):
        inputs = tokenizer(recommendation_train_query[i:i+batch_size], padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings = mean_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = embeddings.detach().cpu().numpy()
        recommendation_train_query_embeddings.append(embeddings)
recommendation_train_query_embeddings = np.concatenate(recommendation_train_query_embeddings)
np.save('recommendation_train_query_embeddings.npy', recommendation_train_query_embeddings)

# Encode recommendation_test_query to recommendation_test_query_embeddings.npy
recommendation_test_query_embeddings = []
with torch.no_grad():
    for i in tqdm(range(0, len(recommendation_test_query), batch_size)):
        inputs = tokenizer(recommendation_test_query[i:i+batch_size], padding=True, truncation=True, return_tensors='pt')
        inputs = {k: v.cuda() for k, v in inputs.items()}
        outputs = model(**inputs)
        embeddings = mean_token_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = embeddings.detach().cpu().numpy()
        recommendation_test_query_embeddings.append(embeddings)
recommendation_test_query_embeddings = np.concatenate(recommendation_test_query_embeddings)
np.save('recommendation_test_query_embeddings.npy', recommendation_test_query_embeddings)
