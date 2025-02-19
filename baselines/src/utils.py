import functools
import torch.nn as nn
from typing import Dict, MutableMapping, Tuple, Union
import yaml
import torch.distributed as dist
import torch
import joblib
import torch.nn.functional as F
import numpy as np
import json
from torch import Tensor
import ast
import sys
import os
from datetime import datetime
from PIL import Image

def vertical_concat_images(images):
    # Get maximum width from all images
    max_width = max(img.width for img in images)

    # Calculate total height
    total_height = sum(img.height for img in images)
    
    # Create a new blank image
    result_image = Image.new('RGB', (max_width, total_height))
    
    # Current height position
    current_height = 0
    
    # Paste images one by one
    for img in images:
        # If image width is less than max width, center it
        if img.width < max_width:
            x_offset = (max_width - img.width) // 2
        else:
            x_offset = 0
            
        # Paste the image
        result_image.paste(img, (x_offset, current_height))
        
        # Update height position
        current_height += img.height
    
    result_image = result_image.resize((1024,1024))
    return result_image

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

def mean_token_pool(last_hidden_states: Tensor,
                   attention_mask: Tensor) -> Tensor:
    """
    Average pooling for non-padding tokens in the sequence
    
    Args:
        last_hidden_states: Tensor of shape (batch_size, sequence_length, hidden_size)
        attention_mask: Binary tensor of shape (batch_size, sequence_length), 1 for actual tokens, 0 for padding
        
    Returns:
        Tensor of shape (batch_size, hidden_size)
    """
    # Expand attention_mask dimensions to match last_hidden_states
    # (batch_size, sequence_length) -> (batch_size, sequence_length, 1)
    attention_mask_expanded = attention_mask.unsqueeze(-1)
    
    # Convert attention mask to float type for calculation
    float_mask = attention_mask_expanded.float()
    
    # Sum the masked hidden states
    sum_hidden_states = torch.sum(last_hidden_states * float_mask, dim=1)
    
    # Calculate the number of actual tokens in each sequence (count of 1s in mask)
    # Add a small value to avoid division by zero
    sequence_lengths = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
    
    # Calculate average
    mean_hidden_states = sum_hidden_states / sequence_lengths.unsqueeze(1)
    
    return mean_hidden_states

def split_string(text, step=20):
    """
    Split string by lines, preserve empty lines, split into list by fixed step size, and maintain original format
    
    Args:
        text (str): Input string
        step (int): Split step size, default is 20
    
    Returns:
        list: Split list, each element maintains original format (including empty lines)
    """
    # Split by lines, preserve all lines (including empty lines)
    lines = text.splitlines()
    
    # Calculate non-empty lines' index
    non_empty_indices = [i for i, line in enumerate(lines) if line.strip()]
    
    # Group non-empty lines' index
    index_groups = [non_empty_indices[i:i+step] for i in range(0, len(non_empty_indices), step)]
    
    result = []
    for group in index_groups:
        # Get current group's first and last index
        start_idx = group[0]
        end_idx = group[-1] + 1
        # Extract complete text segment
        segment = '\n'.join(lines[start_idx:end_idx])
        result.append(segment)
    
    return result


def find_latest_dir_with_subdir(base_path):
    # Specify the target subdirectory path
    target_subdir = "retrieval_lora/new"
    
    # Get all subdirectories
    subdirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Filter directories containing the target subdirectory
    valid_dirs = []
    for d in subdirs:
        full_path = os.path.join(base_path, d, target_subdir)
        if os.path.exists(full_path):
            valid_dirs.append(d)
    
    if not valid_dirs:
        return ''
    
    # Convert directory names to datetime objects for sorting
    # Assuming directory name format is "2025-01-01-14-12-08"
    def parse_dir_date(dir_name):
        try:
            return datetime.strptime(dir_name, "%Y-%m-%d-%H-%M-%S")
        except ValueError:
            return datetime.min
    
    # Get the latest directory
    latest_dir = max(valid_dirs, key=parse_dir_date)
    return os.path.join(base_path, latest_dir)

# lib2to3 is built into Python3, used for automatically converting Python 2 code to Python 3 syntax
try:
    from lib2to3.refactor import RefactoringTool, get_fixers_from_package
    HAS_2TO3 = True
except ImportError:
    HAS_2TO3 = False

def robust_ast_parse(code_str: str):
    """
    Try to parse code_str with Python 3's ast.parse().
    If SyntaxError occurs and lib2to3 is supported, convert the source code to Python 3 and parse again.
    If still fails or lib2to3 is not supported, return None.
    """
    # Step 1: Direct parsing
    try:
        return ast.parse(code_str)
    except SyntaxError:
        pass  # Continue to try 2to3
    
    # If lib2to3 is not available, give up
    if not HAS_2TO3:
        return None
    
    # Step 2: Try to convert with 2to3 and parse again
    try:
        fixers = get_fixers_from_package("lib2to3.fixes")
        refactor_tool = RefactoringTool(fixers)
        # Convert to Python 3 compatible syntax
        try:
            code_str_3 = str(refactor_tool.refactor_string(code_str, name="stdin"))
            return ast.parse(code_str_3)
        except:
            return None
    except SyntaxError:
        return None


def split_code_into_functions(code_str: str):
    """
    Split code_str into multiple blocks, each block contains:
      - All code from the end of the previous function definition to the end of the current function definition.
      - The first function block starts from the beginning of the source code (line index 0).
      - If there are any remaining code lines at the end, they are included in the last block.
    Benefits:
      - No lines are lost between classes, functions, or at the end of the file.
      - If Python2 print causes parse error, it will try to auto-convert and parse.
    Returns a list of strings, each element corresponds to a "function block".
    """
    # First use our encapsulated function for AST parsing
    try:
        tree = robust_ast_parse(code_str)
    except:
        return []
    if tree is None:
        # If parsing fails, return empty list or your desired fallback handling
        return []
    
    # Find all function definition nodes
    func_nodes = [
        n for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    # If no functions, return the entire code as one block
    if not func_nodes:
        return [code_str] if code_str.strip() else []
    
    # Sort by start line number (1-based)
    func_nodes.sort(key=lambda n: n.lineno)
    
    lines = code_str.splitlines()
    total_lines = len(lines)
    
    blocks = []
    prev_end = 0  # Previous block end line index (0-based, left-closed right-open in Python slicing)
    
    for fn in func_nodes:
        # end_lineno is 1-based, requires Python3.8+
        fn_end = fn.end_lineno
        
        # Here we extract source code lines [prev_end, fn_end)
        block_lines = lines[prev_end: fn_end]
        block_code = "\n".join(block_lines)
        blocks.append(block_code)
        
        # Next block starts from fn_end line
        prev_end = fn_end
    
    # If there are remaining lines after the last function, append them to the last block
    if prev_end < total_lines:
        tail_lines = lines[prev_end:]
        blocks[-1] += "\n"
        blocks.append("\n".join(tail_lines))
    
    return blocks

def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]
  

def save_to_json(data, path):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_from_json(path):
    import json
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data
def get_all_reduce_mean(tensor):
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
    tensor = tensor / torch.distributed.get_world_size()
    return tensor

def print_rank_0(msg):
    """Print from process with rank 0 only."""
    if dist.is_available() and dist.is_initialized():
        if dist.get_rank() == 0:
            print(msg, flush=True)
    else:
        print(msg, flush=True)

def print_args(args: MutableMapping[str, object], depth: int = 0):
    """Prints the arguments passed to the script."""
    prefix = "\t" * depth
    for k, v in args.items():
        if isinstance(v, Dict):
            print_rank_0(f"{prefix}{k}:")
            print_args(v, depth + 1)
        else:
            print_rank_0(f"{prefix}{k}: {v}")

def print_trainable_params_stats(model: nn.Module):
    """Prints the number of trainable parameters in the specified model."""
    num_params = sum(p.numel() for p in model.parameters())
    print_rank_0(f"Number of parameters: {num_params/1000000000:.2}B")
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print_rank_0(f"Number of trainable parameters: {trainable_params/1000000}M")
    print_rank_0(f"Ratio of trainable parameters: {trainable_params / num_params:.2%}")
    for para_name, para in model.named_parameters():
        if para.requires_grad==True:
            print_rank_0(f"Trainable parameter: {para_name}")


def get_config(path="config/rellama_config.yaml"):
    with open(path, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config

def rhasattr(obj, attr):
    """A chain-able attribute version of hasattr. For example, to check if
    `obj` has the attribute `foo.bar.baz`, you can use:
        `rhasattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/67303315
    """
    _nested_attrs = attr.split(".")
    _curr_obj = obj
    for _a in _nested_attrs[:-1]:
        if hasattr(_curr_obj, _a):
            _curr_obj = getattr(_curr_obj, _a)
        else:
            return False
    return hasattr(_curr_obj, _nested_attrs[-1])


def rgetattr(obj, attr: str, *args) -> object:
    """A chain-able attribute version of getattr. For example, to get the
    attribute `foo.bar.baz` from `obj`, you can use:
        `rgetattr(obj, "foo.bar.baz")`
    Reference: https://stackoverflow.com/a/31174427
    """

    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def findattr(obj, attrs: Tuple[str]) -> Union[object, None]:
    for attr in attrs:
        if rhasattr(obj, attr):
            return rgetattr(obj, attr)
    raise ValueError(f"Could not find an attribute from `{attrs}` in `{obj}`")
    
def hf_get_decoder_blocks(model: nn.Module) -> Tuple[nn.Module]:
    """Returns the decoder hidden layers of the specified model.
    NOTE: Different model configurations have different hidden layer attribute names.
        - transformer.h: (BloomForCausalLM, GPT2LMHeadModel, GPTJForCausalLM)
        - model.decoder.layers: (OPTForCausalLM)
        - gpt_neox.layers: (GPTNeoXForCausalLM)
        - decoder.block: (T5ForConditionalGeneration)
    """
    hidden_layers_attrs = (
        "h",
        "layers",
        "embed_tokens",
        "model.layers",
        "decoder.layers",
        "transformer.h",
        "transformer.blocks",
        "model.decoder.layers",
        "gpt_neox.layers",
        "decoder.block",
    )
    return findattr(model, hidden_layers_attrs)

def freeze_bottom_causal_layers(model: nn.Module, num_layers_unfrozen: int = 0):
    """Freezes the bottom transformer block layers of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    if num_layers_unfrozen == 0:
        hidden_layers_to_freeze = list(hidden_layers)
    elif num_layers_unfrozen > 0:
        hidden_layers_to_freeze = list(hidden_layers)[:-num_layers_unfrozen]
    else:
        hidden_layers_to_freeze = []
    for layer in hidden_layers_to_freeze:
        layer.requires_grad_(False)

def freeze_non_crossattention_parameters(model: nn.Module, freeze_retrieval_head=False, freeze_lm_head=True):
    """Freezes non cross-attention parameters of the specified model."""
    hidden_layers = hf_get_decoder_blocks(model)
    hidden_layers_to_processing = list(hidden_layers)
    if freeze_lm_head:
        hidden_layers_to_processing.append(findattr(model, ("lm_head", "model.lm_head")))
    if freeze_retrieval_head:
        try:
            hidden_layers_to_processing.append(findattr(model, ("retrieval_head", "model.retrieval_head")))
        except:
            pass
    hidden_layers_to_processing.append(findattr(model, ("model.norm",)))
    for layer in hidden_layers_to_processing:
        for para_name, para in layer.named_parameters():
            if "crossattention" not in para_name:
                para.requires_grad_(False)
            if 'lora' in para_name or 'crossattention' in para_name or 'knowledge_injector' in para_name or "retrieval" in para_name:
                para.requires_grad_(True)

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, threshold=-float('Inf'), filter_value=-np.inf):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
        
    indices_to_remove = logits < threshold
    logits[indices_to_remove] = filter_value
    return logits


def search(index, emb_file, qid_list, outfile, top_k, use_faiss=False):
    search_torch(index, emb_file, qid_list, outfile, top_k)

def search_torch(index, emb_file, qid_list, outfile, top_k):
    q_idx = 0
    with open(outfile, 'w') as out:
        for batch_vec in read_embed(emb_file):
            q_emb_matrix = np.array(batch_vec)
            q_emb_matrix = torch.from_numpy(q_emb_matrix)
            q_emb_matrix = q_emb_matrix.cuda()
            top_k = min(top_k, len(index))
            res_dist, res_p_id = topk_query_passage(q_emb_matrix, index, top_k)
            for i in range(len(q_emb_matrix)):
                qid = qid_list[q_idx]
                for j in range(top_k):
                    pid = res_p_id[i][j]
                    score = res_dist[i][j]
                    out.write('%s\t%s\t%s\t%s\n' % (qid, pid, j+1, score))
                q_idx += 1

from tqdm import tqdm
def read_embed(file_name, dim=768, bs=256):
    if file_name.endswith('npy'):
        i = 0
        emb_np = np.load(file_name)
        with tqdm(total=len(emb_np)//bs+1) as pbar:
            while(i < len(emb_np)):
                vec_list = emb_np[i:i+bs]
                i += bs
                pbar.update(1)
                yield vec_list
    else:
        vec_list = []
        with open(file_name) as inp:
            for line in tqdm(inp):
                data = line.strip()
                vector = [float(item) for item in data.split(' ')]
                assert len(vector) == dim
                vec_list.append(vector)
                if len(vec_list) == bs:
                    yield vec_list
                    vec_list = []
            if vec_list:
                yield vec_list
import torch
def topk_query_passage(query_vector, passage_vector, k):
    """
    Calculate inner product between query vectors and passage vectors, and return indices of top k values

    Args:
        query_vector (torch.Tensor): query vector with shape (batch_size, query_dim)
        passage_vector (torch.Tensor): passage vector with shape (batch_size, passage_dim)
        k (int): number of top k values to return

    Returns:
        torch.Tensor: indices of top k values with shape (batch_size, k)
    """
    # Calculate inner product between query vectors and passage vectors
    scores = torch.matmul(query_vector, passage_vector.t())  # shape: (batch_size, batch_size)

    # Sort each batch and get top k values
    k = min(k, scores.size(1))
    res_dist, res_p_id = torch.topk(scores, k=k, dim=1)  # shape: (batch_size, k)

    return res_dist.cpu().numpy(), res_p_id.cpu().numpy()

def merge(total_part, shift, top, eval_cnts, query_dataset_name, output):
    f_list = []
    for part in range(total_part):
        f0 = open(f'{output}/res.top%d.part%d.step%d.%s' % (top, part, eval_cnts, query_dataset_name))
        f_list.append(f0)

    line_list = []
    for part in range(total_part):
        line = f_list[part].readline()
        line_list.append(line)

    out = open(f'{output}/res.top%d.step%d.%s' % (top, eval_cnts, query_dataset_name), 'w')
    last_q = ''
    ans_list = {}
    while len(line_list):
        cur_list = []
        for line in line_list:
            sub = line.strip().split('\t')
            cur_list.append(sub)

        if last_q == '':
            last_q = cur_list[0][0]
        if cur_list[0][0] != last_q:
            rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
            for i in range(min(top, len(rank))):
                out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
            ans_list = {}
        for i, sub in enumerate(cur_list):
            ans_list[int(sub[1]) + shift*i] = float(sub[-1])
        last_q = cur_list[0][0]

        line_list = []
        for f0 in f_list:
            line = f0.readline()
            sub = line.strip().split('\t')
            if sub[-1]=='':
                continue
            line_list.append(line)

    rank = sorted(ans_list.items(), key = lambda a:a[1], reverse=True)
    for i in range(min(top, len(rank))):
        out.write("%s\t%s\t%s\t%s\n" % (last_q, rank[i][0], i+1, rank[i][1]))
    out.close()


def dict_to_HParams(d):
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_HParams(value)
    return HParams(**d)
class HParams(object):
    """Hyper paramerter"""

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self.__dict__[k] = v

    def __contains__(self, key):
        return key in self.__dict__

    def __getitem__(self, key):
        if key not in self.__dict__:
            raise ValueError('key(%s) not in HParams.' % key)
        return self.__dict__[key]

    def __repr__(self):
        return repr(self.to_dict())

    def __setitem__(self, key, val):
        self.__dict__[key] = val

    @classmethod
    def from_json(cls, json_str):
        """doc"""
        d = json.loads(json_str)
        if type(d) != dict:
            raise ValueError('json object must be dict.')
        return HParams.from_dict(d)

    def get(self, key, default=None):
        """doc"""
        return self.__dict__.get(key, default)

    @classmethod
    def from_dict(cls, d):
        """doc"""
        if type(d) != dict:
            raise ValueError('input must be dict.')
        hp = HParams(**d)
        return hp

    def to_json(self):
        """doc"""
        return json.dumps(self.__dict__)

    def to_dict(self):
        """doc"""
        return self.__dict__
    
    def print_config(self):
        for key,value in self.__dict__.items():
            print(key+":",value)

    def join(self, other):
        """doc"""
        if not isinstance(other, HParams):
            raise ValueError('input must be HParams instance.')
        self.__dict__.update(**other.__dict__)
        return self
