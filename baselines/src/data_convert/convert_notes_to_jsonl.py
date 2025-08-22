import json
import os
import argparse
import pandas as pd
import glob

def convert_collection(args):
    print('Converting collection...')
    
    # Load dataset from parquet files
    notes_path = os.path.join(args.dataset_path, 'notes')
    parquet_files = glob.glob(os.path.join(notes_path, '*.parquet'))
    
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files found in {notes_path}")
    
    print(f"Found {len(parquet_files)} parquet files")
    
    file_index = 0
    output_jsonl_file = None
    total_docs = 0
    
    for parquet_file in sorted(parquet_files):
        print(f"Processing {parquet_file}...")
        df = pd.read_parquet(parquet_file)
        
        for i, row in df.iterrows():
            # Convert row to dictionary and handle NaN values
            doc_dict = {}
            for col in row.index:
                value = row[col]
                # Handle different types of values
                if hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                    # For array-like values (like image_path), convert to list
                    try:
                        doc_dict[col] = value.tolist() if hasattr(value, 'tolist') else list(value)
                    except:
                        doc_dict[col] = str(value)
                elif pd.isna(value):
                    doc_dict[col] = None
                else:
                    doc_dict[col] = value
            
            doc_id = str(row['note_idx'])
            doc_dict['id'] = doc_id
            
            if total_docs % args.max_docs_per_file == 0:
                if output_jsonl_file is not None:
                    output_jsonl_file.close()
                
                output_path = os.path.join(args.output_folder, f'docs{file_index:02d}.jsonl')
                output_jsonl_file = open(output_path, 'w', encoding='utf-8', newline='\n')
                file_index += 1
            
            output_jsonl_file.write(json.dumps(doc_dict, ensure_ascii=False) + '\n')
            
            if total_docs % 100000 == 0:
                print(f'Converted {total_docs:,} docs, writing into file {file_index}')
            
            total_docs += 1
    
    if output_jsonl_file is not None:
        output_jsonl_file.close()
    
    print(f"Total converted documents: {total_docs:,}")

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
