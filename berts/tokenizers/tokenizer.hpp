#pragma once

#include <string>
#include <unordered_set>
#include <vector>
#include "berts/berts.h"
#include "berts/tokenizers/trie.hpp"

namespace berts::tokenizers {

struct context;

//
// new/free context
//

context *new_context(const berts_tokenize_info &cond);

void free_context(context *ctx);

//
// get/set special tokens
//

bert_token_t get_cls_id(context *ctx);

void set_cls_id(context *ctx, bert_token_t id);

bert_token_t get_mask_id(context *ctx);

void set_mask_id(context *ctx, bert_token_t id);

bert_token_t get_pad_id(context *ctx);

void set_pad_id(context *ctx, bert_token_t id);

bert_token_t get_sep_id(context *ctx);

void set_sep_id(context *ctx, bert_token_t id);

bert_token_t get_unk_id(context *ctx);

void set_unk_id(context *ctx, bert_token_t id);

//
// id <-> token conversion
//

std::string id_to_token(context *ctx, bert_token_t token_id);

bert_token_t token_to_id(context *ctx, const std::string &token);

//
// tokens
//

bool add_token(context *ctx, const std::string &token);

bool has_token(context *ctx, const std::string &token);

bool tokenize(const std::string &text,
              const context *ctx,
              std::vector<bert_token_t> &result);

} // namespace berts::tokenizers
