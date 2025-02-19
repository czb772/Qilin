import os
import argparse
from datasets import load_dataset, load_from_disk
from tqdm import tqdm

def convert_queries(args):
    print('Converting queries...')
    
    # Load dataset
    dataset = load_from_disk(args.dataset_path)[args.split]
    
    # Create output file
    output_path = os.path.join(args.output_folder, f'{args.split}_queries.txt')
    
    with open(output_path, 'w', encoding='utf-8', newline='\n') as output_file:
        # Show progress with tqdm
        for item in tqdm(dataset, desc="Processing queries"):
            # Get qid and query
            qid = str(item['search_idx'])  # Ensure id is string type
            query = item['query'].strip()  # Remove possible leading/trailing whitespace
            
            # Write format: qid\tquery\n
            output_file.write(f"{qid}\t{query}\n")
    
    print(f'Converted queries saved to {output_path}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert HuggingFace dataset queries into txt file.')
    parser.add_argument('--dataset_path', required=True, help='Path to HuggingFace dataset or dataset name.')
    parser.add_argument('--output-folder', required=True, help='Output folder.')
    parser.add_argument('--split', default='train', help='Dataset split name (e.g., train, test, validation).')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.output_folder):
        os.makedirs(args.output_folder)
        
    convert_queries(args)
    print('Done!')
