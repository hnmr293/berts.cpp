import argparse
import os
import sys
import struct

import torch
from transformers import (
    BertTokenizer,
    BertModel,
    BertConfig,
)

def xglob(rest: list[str]):
    if sys.platform.lower().startswith('win'):
        import psutil
        p = psutil.Process(os.getpid())
        while p is not None and p.name().lower().startswith('python'):
            p = p.parent()
        if p is None:
            return rest
        if p.name().lower().startswith(('cmd', 'powershell')):
            import glob
            ret = []
            for x in rest:
                ret.extend(glob.glob(x))
            return ret
    return rest


def parse_args():
    ap = argparse.ArgumentParser(prog='test_bert.py')
    ap.add_argument('-i', '--input-model', help='Repo ID of the model', required=True)
    ap.add_argument('-p', '--prompt', default='Hi, I am [MASK] man. How are you?', help='Prompt to test')
    ap.add_argument('--cache-dir', default=None, help='model cache dir')
    ap.add_argument('rest', nargs=argparse.REMAINDER, help='.bin files')

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

    for bin in xglob(args.rest):
        with open(bin, 'rb') as io:
            count = struct.unpack('I', io.read(4))[0]
            data = struct.unpack(f'{count}f', io.read(count * 4))
            data = torch.FloatTensor(data)
            
            print(bin)

            if data.nelement() == expected_pool.nelement():
                # (hidden_dim,)
                data = data.reshape_as(expected_pool)
                norm = torch.linalg.vector_norm(data - expected_pool, dim=-1)
                norm = norm.item()
                assert norm < 0.1, f'{bin}: norm={norm}'
                print(f'  pooled   : {norm}')
            elif data.nelement() == expected_hidden[-1].nelement():
                # (n, hidden_dim)
                data = data.reshape_as(expected_hidden[-1])
                norms = torch.linalg.vector_norm(data - expected_hidden[-1,:,:], dim=-1)
                for i, norm in enumerate(norms):
                    norm = norm.item()
                    assert norm < 0.1, f'{bin}: norm={norm}'
                    print(f'  token {i:>3}: {norm}')
            else:
                raise ValueError(f'unexpected shape = {tuple(data.shape)}')
