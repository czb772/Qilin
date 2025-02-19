import json
import os
import argparse
from datasets import load_dataset, load_from_disk

def convert_collection(args):
    print('Converting collection...')
    
    # Load dataset
    dataset = load_from_disk(args.dataset_path)['notes']
    
    file_index = 0
    output_jsonl_file = None
    
    for i, item in enumerate(dataset):
        # Merge title and content as document content
        doc_text = item['note_title'] + " " + item['note_content']
        doc_id = str(item['note_idx'])
        
        if i % args.max_docs_per_file == 0:
            if output_jsonl_file is not None:
                output_jsonl_file.close()
            
            output_path = os.path.join(args.output_folder, f'docs{file_index:02d}.json')
            output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
            file_index += 1
        
        output_dict = {'id': doc_id, 'contents': doc_text}
        output_jsonl_file.write(json.dumps(output_dict, ensure_ascii=False) + '\n')
        
        if i % 100000 == 0:
            print(f'Converted {i:,} docs, writing into file {file_index}')
    
    if output_jsonl_file is not None:
        output_jsonl_file.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HuggingFace dataset into jsonl files for Anserini.')
    parser.add_argument('--dataset_path', required=True, help='Path to HuggingFace dataset or dataset name.')
    parser.add_argument('--output-folder', required=True, help='Output folder.')
    parser.add_argument('--max-docs-per-file', default=1000000, type=int,
                      help='Maximum number of documents in each jsonl file.')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    convert_collection(args)
    print('Done!')
