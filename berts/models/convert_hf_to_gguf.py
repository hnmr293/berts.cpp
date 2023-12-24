import argparse
import os
import json

import numpy as np
import gguf
from transformers import (
    #AutoTokenizer,
    #AutoModel,
    BertTokenizer,
    BertModel,
    BertConfig,
)

def KEY(s: str):
    return 'berts.bert.' + s

def KEY_N(s: str):
    return 'berts.bert.{n}.' + s

# ggml_type
GGML_TYPE_I8 = 16
GGML_TYPE_I16 = 17
GGML_TYPE_I32 = 18
GGML_TYPE_COUNT = 19

# hparams
BERTS_KEY_HPARAM_BERT_TYPE = 'berts.bert_type'
BERTS_KEY_HPARAM_VOCAB_SIZE = 'berts.vocab_size'
BERTS_KEY_HPARAM_HIDDEN_DIM = 'berts.hidden_dim'
BERTS_KEY_HPARAM_N_LAYERS = 'berts.n_layers'
BERTS_KEY_HPARAM_ATTN_HEADS = 'berts.attn_heads'
BERTS_KEY_HPARAM_MAX_TOKENS = 'berts.max_token'
BERTS_KEY_HPARAM_INTERMEDIATE_DIM = 'berts.intermediate_dim'
BERTS_KEY_HPARAM_HIDDEN_ACT = 'berts.hidden_act'

# tensor keys
BERTS_KEY_BERT_VOCAB_SIZE = KEY('vocab_size')
BERTS_KEY_BERT_VOCAB_DATA = KEY('vocab_data')
BERTS_KEY_BERT_EMB_TOKEN = KEY('token_embedding')
BERTS_KEY_BERT_EMB_SEGM = KEY('segment_embedding')
BERTS_KEY_BERT_EMB_POS = KEY('position_embedding')
BERTS_KEY_BERT_LN_W = KEY('ln_w')
BERTS_KEY_BERT_LN_B = KEY('ln_b')
BERTS_KEY_BERT_ENC_N_Q_W = KEY_N('q_w')
BERTS_KEY_BERT_ENC_N_Q_B = KEY_N('q_b')
BERTS_KEY_BERT_ENC_N_K_W = KEY_N('k_w')
BERTS_KEY_BERT_ENC_N_K_B = KEY_N('k_b')
BERTS_KEY_BERT_ENC_N_V_W = KEY_N('v_w')
BERTS_KEY_BERT_ENC_N_V_B = KEY_N('v_b')
BERTS_KEY_BERT_ENC_N_FF_W = KEY_N('ff_w')
BERTS_KEY_BERT_ENC_N_FF_B = KEY_N('ff_b')
BERTS_KEY_BERT_ENC_N_LN_FF_W = KEY_N('ln_ff_w')
BERTS_KEY_BERT_ENC_N_LN_FF_B = KEY_N('ln_ff_b')
BERTS_KEY_BERT_ENC_N_I_W = KEY_N('i_w')
BERTS_KEY_BERT_ENC_N_I_B = KEY_N('i_b')
BERTS_KEY_BERT_ENC_N_O_W = KEY_N('o_w')
BERTS_KEY_BERT_ENC_N_O_B = KEY_N('o_b')
BERTS_KEY_BERT_ENC_N_LN_OUT_W = KEY_N('ln_out_w')
BERTS_KEY_BERT_ENC_N_LN_OUT_B = KEY_N('ln_out_b')

def parse_args():
    ap = argparse.ArgumentParser(prog='convert_hf_to_gguf.py')
    ap.add_argument('-i', '--input-model', help='Repo ID of the model', required=True)
    ap.add_argument('-o', '--output-path', help='Path to save GGUF file', required=True)
    ap.add_argument('--use-f32', action='store_true', default=False, help='Use f32 instead of f16')
    ap.add_argument('--cache-dir', default=None, help='model cache dir')

    args = ap.parse_args()
    return args

def load_diffusers(repo_id: str, cache_dir: str|None) -> tuple[BertTokenizer, BertModel]:
    tokenizer = BertTokenizer.from_pretrained(repo_id, cache_dir=cache_dir)
    model = BertModel.from_pretrained(repo_id, cache_dir=cache_dir)
    return tokenizer, model

def convert(repo_id: str, cache_dir: str|None, output_path: str):
    print('start quantization')
    
    tokenizer, model = load_diffusers(repo_id, cache_dir)
    config: BertConfig = model.config

    print(f'model loaded: {repo_id}')
    
    w = gguf.GGUFWriter(output_path, arch='BERT')

    w.add_name(repo_id)
    w.add_custom_alignment(w.data_alignment) # if omitted, writer never writes "general.alignment"

    print(f'''
[hparams]
  vocab_size = {config.vocab_size}
  hidden_dim = {config.hidden_size}
  n_layers   = {config.num_hidden_layers}
  attn_heads = {config.num_attention_heads}
  max_token  = {config.max_position_embeddings}
  interm_dim = {config.intermediate_size}
  hidden_act = {config.hidden_act}
'''.strip())
    
    # hparams
    w.add_uint32(BERTS_KEY_HPARAM_BERT_TYPE, 0) # BERT
    w.add_uint32(BERTS_KEY_HPARAM_VOCAB_SIZE, config.vocab_size)
    w.add_uint32(BERTS_KEY_HPARAM_HIDDEN_DIM, config.hidden_size)
    w.add_uint32(BERTS_KEY_HPARAM_N_LAYERS, config.num_hidden_layers)
    w.add_uint32(BERTS_KEY_HPARAM_ATTN_HEADS, config.num_attention_heads)
    w.add_uint32(BERTS_KEY_HPARAM_MAX_TOKENS, config.max_position_embeddings)
    w.add_uint32(BERTS_KEY_HPARAM_INTERMEDIATE_DIM, config.intermediate_size)
    w.add_uint32(BERTS_KEY_HPARAM_HIDDEN_ACT, 0) # GeLU
    assert model.config.hidden_act == 'gelu'
    
    ftype = 1 # f16
    if args.use_f32:
        ftype = 0 # f32
    ftype_str = ['f32', 'f16'][ftype]
    
    w.add_file_type(ftype)

    print('writing vocab...')

    vocab_dict = tokenizer.get_vocab()
    tokens = [None] * config.vocab_size
    for token, id in vocab_dict.items():
        assert id <= len(tokens), f'id={id}, len(tokens)={len(tokens)}'
        assert tokens[id] is None, f'id={id}, tokens[id]={tokens[id]}'
        tokens[id] = token
    unused = 0
    token_lengths = []
    token_bytes = []
    for token in tokens:
        if token is None:
            token = f'[unused{unused}]'
            unused += 1
        bytes = token.encode('utf-8')
        token_lengths.append(len(bytes))
        token_bytes.append(bytes)
    token_lengths = np.array(token_lengths, dtype=np.int32)
    token_bytes = np.frombuffer(b''.join(token_bytes), dtype=np.int8)

    w.add_tensor(BERTS_KEY_BERT_VOCAB_SIZE, token_lengths, raw_dtype=GGML_TYPE_I32)
    w.add_tensor(BERTS_KEY_BERT_VOCAB_DATA, token_bytes, raw_dtype=GGML_TYPE_I8)

    print(f'''
[tokens]
  total  = {len(tokens)}
  used   = {len(tokens) - unused}
  unused = {unused}
'''.strip())

    print('converting...')
    
    total_size_org = 0
    total_size_new = 0

    for key, tensor in model.state_dict().items():
        tensor: np.ndarray = tensor.squeeze().cpu().numpy()
        n_dims = tensor.ndim
        size_org = tensor.nbytes
        
        q = False
        if ftype != 0 and n_dims == 2:
            if key[-7:] == '.weight':
                q = True
        
        if q:
            tensor = tensor.astype(np.float16)
        else:
            tensor = tensor.astype(np.float32)
        
        size_new = tensor.nbytes

        total_size_org += size_org
        total_size_new += size_new

        w.add_tensor('berts.bert.' + key, tensor)

        print(f'''
{key}:
  quantized = {q}
  n_dims = {n_dims}
  size = {size_org/1024:.1f} KiB -> {size_new/1024:.1f} KiB
'''.strip())
    
    print(f'''
{"="*40}
original size  = {total_size_org/1024/1024:.1f} MiB
quantized size = {total_size_new/1024/1024:.1f} MiB
{"="*40}
'''.strip())
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()

    print('quantization completed')

if __name__ == '__main__':
    args = parse_args()
    repo_id = args.input_model
    cache_dir = args.cache_dir
    output_path = args.output_path
    convert(repo_id, cache_dir, output_path)
