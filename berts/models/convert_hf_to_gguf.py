import argparse
import os
import re

import numpy as np
from torch.nn import Module
import gguf
from transformers import *
import transformers.models.bert.modeling_bert as BERT
import transformers.models.roberta.modeling_roberta as RoBERTa
import transformers.models.deberta.modeling_deberta as DeBERTa



def KEY(s: str):
    return 'berts.bert.' + s

def KEY_N(s: str):
    return 'berts.bert.{n}.' + s

# ggml_type
GGML_TYPE_I8 = 16
GGML_TYPE_I16 = 17
GGML_TYPE_I32 = 18
GGML_TYPE_COUNT = 19

def load_keys(path: str) -> dict[str, str]:
    with open(path, 'r', encoding='utf-8') as io:
        lines = io.readlines()
    
    r = re.compile(r'^#define\s+(\w+)\s+"([^"]+)"$')
    result = dict()
    for line in lines:
        line = line.strip()
        if not line.startswith('#define'):
            continue
        m = r.match(line)
        assert m is not None, line
        key, val = m.group(1, 2)
        result[key] = val
    return result

def parse_args():
    ap = argparse.ArgumentParser(prog='convert_hf_to_gguf.py')
    ap.add_argument('-i', '--input-model', help='Repo ID of the model', required=True)
    ap.add_argument('-o', '--output-path', help='Path to save GGUF file', required=True)
    ap.add_argument('--use-f32', action='store_true', default=False, help='Use f32 instead of f16')
    ap.add_argument('--cache-dir', default=None, help='model cache dir')

    args = ap.parse_args()
    return args

def load_diffusers(repo_id: str, cache_dir: str|None) -> tuple[str, PreTrainedTokenizer, PreTrainedModel, Module]:
    tokenizer = AutoTokenizer.from_pretrained(repo_id, cache_dir=cache_dir, use_fast=False)
    model = AutoModel.from_pretrained(repo_id, cache_dir=cache_dir)
    lm = AutoModelForMaskedLM.from_pretrained(repo_id, cache_dir=cache_dir)
    
    if isinstance(lm, BertForMaskedLM):
        type = 'BERT'
        lm = lm.cls
    elif isinstance(lm, RobertaForMaskedLM):
        type = 'RoBERTa'
        lm = lm.lm_head
    elif isinstance(lm, DebertaForMaskedLM):
        type = 'DeBERTa'
        lm = lm.cls
    else:
        raise ValueError(f'unsupported class: {model.__class__.__name__}')

    return type, tokenizer, model, lm

def write_bert(w: gguf.GGUFWriter, tokenizer: BertTokenizer, model: BertModel, K: dict[str, str]):
    config: BertConfig = model.config
    
    # hparams
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    max_position_embeddings = config.max_position_embeddings
    intermediate_size = config.intermediate_size
    hidden_act = config.hidden_act
    layer_norm_eps = config.layer_norm_eps
    type_vocab_size = config.type_vocab_size
    initializer_range = config.initializer_range

    print(f'''
[hparams]
  arch       = BERT
  vocab_size = {vocab_size}
  hidden_dim = {hidden_size}
  n_layers   = {num_hidden_layers}
  attn_heads = {num_attention_heads}
  max_token  = {max_position_embeddings}
  interm_dim = {intermediate_size}
  hidden_act = {hidden_act}
  segments   = {type_vocab_size}
  eps        = {layer_norm_eps}
  init_range = {initializer_range}
'''.strip())
    
    w.add_uint32(K['BERTS_KEY_HPARAM_BERT_TYPE'], 0) # BERTS_TYPE_BERT
    w.add_uint32(K['BERTS_KEY_HPARAM_VOCAB_SIZE'], vocab_size)
    w.add_uint32(K['BERTS_KEY_HPARAM_HIDDEN_DIM'], hidden_size)
    w.add_uint32(K['BERTS_KEY_HPARAM_N_LAYERS'], num_hidden_layers)
    w.add_uint32(K['BERTS_KEY_HPARAM_ATTN_HEADS'], num_attention_heads)
    w.add_uint32(K['BERTS_KEY_HPARAM_MAX_TOKENS'], max_position_embeddings)
    w.add_uint32(K['BERTS_KEY_HPARAM_INTERMEDIATE_DIM'], intermediate_size)
    w.add_uint32(K['BERTS_KEY_HPARAM_HIDDEN_ACT'], 0) # GeLU
    assert model.config.hidden_act == 'gelu'
    w.add_float64(K['BERTS_KEY_HPARAM_LN_EPS'], layer_norm_eps)
    w.add_uint32(K['BERTS_KEY_HPARAM_SEGM_COUNT'], type_vocab_size)
    w.add_float64(K['BERTS_KEY_HPARAM_INIT_RANGE'], initializer_range)

    
    # special tokens
    cls_token_id = tokenizer.cls_token_id
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    unk_token_id = tokenizer.unk_token_id

    print(f'''
[special tokens]
  cls_id  = {cls_token_id} {tokenizer.cls_token}
  mask_id = {mask_token_id} {tokenizer.mask_token}
  pad_id  = {pad_token_id} {tokenizer.pad_token}
  sep_id  = {sep_token_id} {tokenizer.sep_token}
  unk_id  = {unk_token_id} {tokenizer.unk_token}
'''.strip())
    
    if cls_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_CLS_ID'], cls_token_id)
    if mask_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_MASK_ID'], mask_token_id)
    if pad_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_PAD_ID'], pad_token_id)
    if sep_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_SEP_ID'], sep_token_id)
    if unk_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_UNK_ID'], unk_token_id)

    # tokenizer params
    def with_basic(tokenizer: PreTrainedTokenizer, attr: str, default = None):
        if hasattr(tokenizer, 'basic_tokenizer'):
            return getattr(tokenizer.basic_tokenizer, attr, default)
        return default
    
    do_lower_case = tokenizer.do_lower_case
    do_basic_tokenize = tokenizer.do_basic_tokenize
    tokenize_chinese_chars = with_basic(tokenizer, 'tokenize_chinese_chars')
    never_split = with_basic(tokenizer, 'never_split')
    strip_accents = with_basic(tokenizer, 'strip_accents')

    print(f'''
[tokenizer params]
  do_lower_case = {do_lower_case}
  do_basic_tokenize = {do_basic_tokenize}
  tokenize_chinese_chars = {tokenize_chinese_chars}
  never_split = {never_split}
  strip_accents = {strip_accents}
'''.strip())
    
    if do_lower_case is not None:
        w.add_bool(K['BERTS_KEY_TOKENIZER_DO_LOWER_CASE'], do_lower_case)
    if do_basic_tokenize is not None:
        w.add_bool(K['BERTS_KEY_TOKENIZER_DO_BASIC_TOKENIZE'], do_basic_tokenize)
    if tokenize_chinese_chars is not None:
        w.add_bool(K['BERTS_KEY_TOKENIZER_CHINESE_CHARS'], tokenize_chinese_chars)
    # never split ???
    if never_split is not None:
        if len(never_split) != 0:
            raise RuntimeError(f'never_split: {never_split}')
            w.add_bool(K['BERTS_KEY_TOKENIZER_NEVER_SPLIT'], tokenizer.basic_tokenizer.never_split)
    if strip_accents is not None:
        w.add_bool(K['BERTS_KEY_TOKENIZER_STRIP_ACCENT'], strip_accents)


def write_roberta(w: gguf.GGUFWriter, tokenizer: RobertaTokenizer, model: RobertaModel, K: dict[str, str]):
    config: RobertaConfig = model.config
    
    # hparams
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_hidden_layers = config.num_hidden_layers
    num_attention_heads = config.num_attention_heads
    max_position_embeddings = config.max_position_embeddings
    intermediate_size = config.intermediate_size
    hidden_act = config.hidden_act
    layer_norm_eps = config.layer_norm_eps
    type_vocab_size = config.type_vocab_size
    initializer_range = config.initializer_range
    
    print(f'''
[hparams]
  arch       = RoBERTa
  vocab_size = {vocab_size}
  hidden_dim = {hidden_size}
  n_layers   = {num_hidden_layers}
  attn_heads = {num_attention_heads}
  max_token  = {max_position_embeddings}
  interm_dim = {intermediate_size}
  hidden_act = {hidden_act}
  segments   = {type_vocab_size}
  eps        = {layer_norm_eps}
  init_range = {initializer_range}
'''.strip())
    
    w.add_uint32(K['BERTS_KEY_HPARAM_BERT_TYPE'], 1) # BERTS_TYPE_ROBERTA
    w.add_uint32(K['BERTS_KEY_HPARAM_VOCAB_SIZE'], vocab_size)
    w.add_uint32(K['BERTS_KEY_HPARAM_HIDDEN_DIM'], hidden_size)
    w.add_uint32(K['BERTS_KEY_HPARAM_N_LAYERS'], num_hidden_layers)
    w.add_uint32(K['BERTS_KEY_HPARAM_ATTN_HEADS'], num_attention_heads)
    w.add_uint32(K['BERTS_KEY_HPARAM_MAX_TOKENS'], max_position_embeddings)
    w.add_uint32(K['BERTS_KEY_HPARAM_INTERMEDIATE_DIM'], intermediate_size)
    match hidden_act.lower():
        case 'gelu':
            hidden_act_type = 0
        case 'relu':
            hidden_act_type = 1
        case 'silu':
            hidden_act_type = 2
        case 'gelu_new':
            hidden_act_type = 3
        case None:
            # use default value, GeLU
            hidden_act_type = 0
        case _:
            raise ValueError(f'unknown hidden act: {hidden_act}')
    w.add_uint32(K['BERTS_KEY_HPARAM_HIDDEN_ACT'], hidden_act_type)
    w.add_float64(K['BERTS_KEY_HPARAM_LN_EPS'], layer_norm_eps)
    w.add_uint32(K['BERTS_KEY_HPARAM_SEGM_COUNT'], type_vocab_size)
    w.add_float64(K['BERTS_KEY_HPARAM_INIT_RANGE'], initializer_range)

    
    # special tokens
    cls_token_id = tokenizer.cls_token_id
    mask_token_id = tokenizer.mask_token_id
    pad_token_id = tokenizer.pad_token_id
    sep_token_id = tokenizer.sep_token_id
    unk_token_id = tokenizer.unk_token_id
    bos_token_id = tokenizer.bos_token_id
    eos_token_id = tokenizer.eos_token_id

    print(f'''
[special tokens]
  cls_id  = {cls_token_id} {tokenizer.cls_token}
  mask_id = {mask_token_id} {tokenizer.mask_token}
  pad_id  = {pad_token_id} {tokenizer.pad_token}
  sep_id  = {sep_token_id} {tokenizer.sep_token}
  unk_id  = {unk_token_id} {tokenizer.unk_token}
  bos_id  = {bos_token_id} {tokenizer.bos_token}
  eos_id  = {eos_token_id} {tokenizer.eos_token}
'''.strip())

    if cls_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_CLS_ID'], cls_token_id)
    if mask_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_MASK_ID'], mask_token_id)
    if pad_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_PAD_ID'], pad_token_id)
    if sep_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_SEP_ID'], sep_token_id)
    if unk_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_UNK_ID'], unk_token_id)
    if bos_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_BOS_ID'], bos_token_id)
    if eos_token_id is not None:
        w.add_uint32(K['BERTS_KEY_TOKENIZER_EOS_ID'], eos_token_id)

    # tokenizer params
    # do nothing
    

def write_vocab(w: gguf.GGUFWriter, tokenizer: PreTrainedTokenizer, model: PreTrainedModel, K: dict):
    print('writing vocab...')

    vocab_dict = tokenizer.get_vocab()
    tokens = [None] * model.config.vocab_size
    for token, id in vocab_dict.items():
        assert id <= len(tokens), f'id={id}, len(tokens)={len(tokens)}'
        assert tokens[id] is None, f'id={id}, tokens[id]={tokens[id]}'
        tokens[id] = token
    
    unused = 0
    unused_index = 0
    token_lengths = []
    token_bytes = []
    for token in tokens:
        # make unused tokens
        if token is None:
            token = f'[unused{unused_index}]'
            while token in tokens:
                unused_index += 1
                token = f'[unused{unused_index}]'
            unused += 1
            unused_index += 1
        
        # build token bytes
        bytes = token.encode('utf-8')
        n_bytes = len(bytes)
        if n_bytes == 256:
            n_bytes = 0
        elif 256 < n_bytes:
            raise ValueError(f'too long token: {token} ({n_bytes})')
        token_lengths.append(n_bytes)
        token_bytes.append(bytes)
    
    token_lengths = np.array(token_lengths, dtype=np.uint8)
    token_bytes = b''.join(token_bytes)
    token_bytes = np.frombuffer(token_bytes, dtype=np.int8)
    
    w.add_tensor(K['BERTS_KEY_ALL_VOCAB_SIZE'], token_lengths, raw_dtype=GGML_TYPE_I8)
    w.add_tensor(K['BERTS_KEY_ALL_VOCAB_DATA'], token_bytes, raw_dtype=GGML_TYPE_I8)

    print(f'''
[tokens]
  total  = {len(tokens)}
  used   = {len(tokens) - unused}
  unused = {unused}
'''.strip())


def write_merge(w: gguf.GGUFWriter, vocab: dict[str, int], merges: dict[tuple[str,str], int], K: dict):
    print('writing bpe merges...')
    
    #    [ (token0a, token0b) -> rank0, (token1a, token1b) -> rank1, ... ]
    # -> [ id0a id0b rank0 id1a id1b rank1 ... ]
    merge_data = []
    for (token0, token1), rank in merges.items():
        id0 = vocab.get(token0, None)
        id1 = vocab.get(token1, None)
        if id0 is None:
            raise ValueError(f'{token0} is not found in vocab')
        if id1 is None:
            raise ValueError(f'{token1} is not found in vocab')
        merge_data.extend([id0, id1, rank])
    
    merge_data = np.array(merge_data, dtype=np.int32)
    
    w.add_tensor(K['BERTS_KEY_ALL_MERGE_DATA'], merge_data, raw_dtype=GGML_TYPE_I32)

    print(f'''
[merges]
  total = {len(merges)}
'''.strip())


def convert(repo_id: str, cache_dir: str|None, output_path: str):
    print('start quantization')
    
    type, tokenizer, model, lm = load_diffusers(repo_id, cache_dir)

    print(f'model loaded: {repo_id}')
    
    match model.__class__.__name__:
        case 'BertModel':
            w = gguf.GGUFWriter(output_path, arch='BERT')
        case 'RobertaModel':
            w = gguf.GGUFWriter(output_path, arch='RoBERTa')
        case _:
            raise ValueError('must not happen')

    w.add_name(repo_id)
    w.add_custom_alignment(w.data_alignment) # if omitted, writer never writes "general.alignment"

    K = load_keys(os.path.dirname(__file__) + '/keys.h')

    match model.__class__.__name__:
        case 'BertModel':
            write_bert(w, tokenizer, model, K)
        case 'RobertaModel':
            write_roberta(w, tokenizer, model, K)
        case _:
            raise ValueError('must not happen')

    ftype = 1 # f16
    if args.use_f32:
        ftype = 0 # f32
    #ftype_str = ['f32', 'f16'][ftype]
    
    w.add_file_type(ftype)

    write_vocab(w, tokenizer, model, K)

    if hasattr(tokenizer, 'bpe_ranks'):
        # tokenizer has merges.txt
        write_merge(w, tokenizer.get_vocab(), tokenizer.bpe_ranks, K)

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
    
    
    lm_skip_keys = set([
        'predictions.bias',
    ])

    RoBERTa_to_BERT = {
        'bias': 'predictions.bias',
        'dense.weight': 'predictions.transform.dense.weight',
        'dense.bias': 'predictions.transform.dense.bias',
        'layer_norm.weight': 'predictions.transform.LayerNorm.weight',
        'layer_norm.bias': 'predictions.transform.LayerNorm.bias',
        'decoder.weight': 'predictions.decoder.weight',
        'decoder.bias': 'predictions.decoder.bias',
    }
    
    for key, tensor in lm.state_dict().items():
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

        if type == 'RoBERTa':
            key = RoBERTa_to_BERT.get(key, key)
        
        if key in lm_skip_keys:
            continue

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
