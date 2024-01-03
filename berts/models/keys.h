#pragma once

//
// general variables of BERT
//

//
// config.json
//

// type: u32 [bert_type]
#define BERTS_KEY_HPARAM_BERT_TYPE "berts.bert_type"

// type: u32
#define BERTS_KEY_HPARAM_VOCAB_SIZE "berts.vocab_size"

// type: u32
#define BERTS_KEY_HPARAM_HIDDEN_DIM "berts.hidden_dim"

// type: u32
#define BERTS_KEY_HPARAM_N_LAYERS "berts.n_layers"

// type: u32
#define BERTS_KEY_HPARAM_ATTN_HEADS "berts.attn_heads"

// type: u32
#define BERTS_KEY_HPARAM_MAX_TOKENS "berts.max_token"

// type: u32
#define BERTS_KEY_HPARAM_INTERMEDIATE_DIM "berts.intermediate_dim"

// type: u32 [hidden_act]
#define BERTS_KEY_HPARAM_HIDDEN_ACT "berts.hidden_act"

// type: f64 *optional, defaults to `1e-12`
#define BERTS_KEY_HPARAM_LN_EPS "berts.layer_norm_eps"

// type: u32 *optional, defaults to `2`
#define BERTS_KEY_HPARAM_SEGM_COUNT "berts.segments"

// type: f64 *optional, defaults to `0.02`
#define BERTS_KEY_HPARAM_INIT_RANGE "berts.initializer_range"

//
// tokenizer_config.json
//

// type: u32 *optional, defaults to id of [CLS] or <s>
#define BERTS_KEY_TOKENIZER_CLS_ID "berts.cls_id"

// type: u32 *optional, defaults to id of [MASK] or <mask>
#define BERTS_KEY_TOKENIZER_MASK_ID "berts.mask_id"

// type: u32 *optional, defaults to id of [PAD] or <pad>
#define BERTS_KEY_TOKENIZER_PAD_ID "berts.pad_id"

// type: u32 *optional, defaults to id of [SEP] or </s>
#define BERTS_KEY_TOKENIZER_SEP_ID "berts.sep_id"

// type: u32 *optional, defaults to id of [UNK] or <unk>
#define BERTS_KEY_TOKENIZER_UNK_ID "berts.unk_id"

// type: u32 *optional, defaults to id of cls token
#define BERTS_KEY_TOKENIZER_BOS_ID "berts.bos_id"

// type: u32 *optional, defaults to id of sep token
#define BERTS_KEY_TOKENIZER_EOS_ID "berts.eos_id"

// type: bool *optional, defaults to `true`
#define BERTS_KEY_TOKENIZER_DO_LOWER_CASE "berts.do_lower_case"

// type: bool *optional, defaults to `true`
#define BERTS_KEY_TOKENIZER_DO_BASIC_TOKENIZE "berts.do_basic_tokenize"

// type: [i32] *optional
#define BERTS_KEY_TOKENIZER_NEVER_SPLIT "berts.never_split"

// type: bool *optional, defaults to `true`
#define BERTS_KEY_TOKENIZER_CHINESE_CHARS "berts.tokenize_chinese_chars"

// type: bool *optional, defaults to the value of `BERTS_KEY_TOKENIZER_DO_LOWER_CASE`
#define BERTS_KEY_TOKENIZER_STRIP_ACCENT "berts.strip_accent"

//
// other global vars
//

// type: [i32]
#define BERTS_KEY_ALL_VOCAB_SIZE "berts.vocab.size"

// type: [u8]
#define BERTS_KEY_ALL_VOCAB_DATA "berts.vocab.data"
