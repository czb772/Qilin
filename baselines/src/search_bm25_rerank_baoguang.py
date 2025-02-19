from datasets import load_from_disk
from evaluator import BM25Evaluator
qilin = load_from_disk('datasets/qilin_data_training_0210_v3/')

search_test = qilin['search_test']
search_test = search_test.select(range(1000))

evaluator = BM25Evaluator(search_test, qilin['notes'], '../datasets/search.test.qrels.csv', top_k=100)

metrics = evaluator.evaluate()
print(metrics)
# {'MRR@10': 0.3387956349206345, 'MAP@10': 0.2398558287666916, 'Recall@10': 0.5898117514662637, 'Precision@10': 0.19619999999999824, 'MRR@100': 0.34670575001254395, 'MAP@100': 0.31391315638201395, 'Recall@100': 0.9950208920362356, 'Precision@100': 0.05342000000000021}
