import argparse
import os
import glob
import struct

import torch
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
)

def parse_args():
    ap = argparse.ArgumentParser(prog='test_infer.py')
    ap.add_argument('-i', '--input-model', help='Repo ID of the model', required=True)
    ap.add_argument('-p', '--prompt', default='Hi, I am [MASK] man. How are you?', help='Prompt to test')
    ap.add_argument('--cache-dir', default=None, help='model cache dir')

    args = ap.parse_args()
    return args

def load_diffusers(repo_id: str, cache_dir: str|None) -> tuple[BertTokenizer, BertModel]:
    tokenizer = BertTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    model = BertModel.from_pretrained(repo_id, cache_dir=cache_dir)
    return tokenizer, model

def get_diffuser_result(repo_id: str, prompt: str, cache_dir: str|None = None):
    print(f'prompt = {prompt}')
    
    tokenizer, model = load_diffusers(repo_id, cache_dir)
    
    inp = tokenizer(prompt, return_tensors='pt')
    print(f'ids = {inp.input_ids}')
    
    out = model(**inp, output_hidden_states=True, return_dict=True)
    pool = out.pooler_output
    hidden = out.hidden_states

    pool = pool.cpu().to(dtype=torch.float32).squeeze(0)
    hidden = torch.concat(hidden).cpu().to(dtype=torch.float32).squeeze(1)
    
    return pool, hidden

if __name__ == '__main__':
    args = parse_args()
    repo_id = args.input_model
    cache_dir = args.cache_dir
    prompt = args.prompt

    expected_pool, expected_hidden \
        = get_diffuser_result(repo_id, prompt, cache_dir=cache_dir)

    for bin in glob.glob(f'{os.path.dirname(__file__)}/test_eval_*.bin'):
        with open(bin, 'rb') as io:
            count = struct.unpack('I', io.read(4))[0]
            data = struct.unpack(f'{count}f', io.read(count * 4))
            data = torch.FloatTensor(data).reshape((-1, 768)).squeeze(0)
            
            print(bin)

            if data.shape == expected_pool.shape:
                # (hidden_dim,)
                norm = torch.linalg.vector_norm(data - expected_pool, dim=-1)
                print(f'  pooled   : {norm.item()}')
            elif data.shape == expected_hidden[-1].shape:
                # (n, hidden_dim)
                norm = torch.linalg.vector_norm(data - expected_hidden[-1,:,:], dim=-1)
                for i, n in enumerate(norm):
                    print(f'  token {i:>3}: {n.item()}')
            else:
                raise ValueError(f'unexpected shape = {tuple(data.shape)}')
