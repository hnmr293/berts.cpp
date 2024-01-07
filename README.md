# berts.cpp

[ggml](https://github.com/ggerganov/ggml) implementation of BERT.

## Examples

### converting `diffusers` model to `gguf` format

```
$ .venv/bin/activate
$ python berts/models/convert_hf_to_gguf.py \
    -i "bert-base-cased" \
    -o "bert-base-cased-f32.gguf" \
    --use-f32 
```

### quantization

```
$ bers/quant bert-base-cased-f32.gguf Q8_0 bert-base-cased-q8_0.gguf
> original size  = 525014746 (500.7 MiB)
> quantized size = 140064442 (133.6 MiB)
```

### filling mask

```
$ berts/fill_mask bert-base-cased-q8_0.gguf 3 "Hello I'm a [MASK] model."
> prompt = Hello I'm a [MASK] model.
> token id = 101 8667 146 112 182 170 103 2235 119 102
> 0: fashion (4633); p = 0.0899522
> 1: new     (1207); p = 0.0637747
> 2: male    (2581); p = 0.0619739
```

## How to build

### Dependency

- [`ICU`](https://unicode-org.github.io/icu/)
- [`{fmt}`](https://fmt.dev/latest/index.html) (if you need)

```bash
$ git clone --depth 1 https://github.com/hnmr293/berts.cpp --recurse-submodules --shallow-submodules
$ cd berts.cpp
$ python -m venv .venv --prompt "berts.cpp"
$ .venv/bin/activate
$ pip install -r requirements.txt
$ make
```

### Variables for `make`

pass them to `make` as environment variables with `Variable=Value make` format.

|Name|Effect|Example
|---  |---   |---  |
|`BERTS_DEBUG`|Enable debugging|`BERTS_DEBUG=1`|
|`BERTS_ICU_INCLUDE`|Specify include path of [ICU4C](https://unicode-org.github.io/icu/userguide/icu4c/) headers relative to `berts` directory|`BERTS_ICU_INCLUDE='C:/Program Files (x86)/Windows Kits/10/Include/10.0.22000.0/um'`|
|`BERTS_ICU_LIB`|Specify search path of [ICU4C](https://unicode-org.github.io/icu/userguide/icu4c/) libs relative to `berts` directory|`BERTS_ICU_LIB='C:/Program Files (x86)/Windows Kits/10/Lib/10.0.22000.0/um/x64'`|
|`BERTS_USE_FMTLIB_FMT`|Use [{fmt} lib](https://github.com/fmtlib/fmt) instead of std::format|`BERTS_USE_FMTLIB_FMT=1`|
|`BERTS_FMTLIB_FMT_INCLUDE`|Specify include path of the {fmt} header relative to `berts` directory (passed to `-I`)|`BERTS_FMTLIB_FMT_INCLUDE="../fmt/include"`|
|`BERTS_FMTLIB_FMT_LIB`|Specify search path for `libfmt.a` relative to `berts` directory (passed to `-L`)|`BERTS_FMTLIB_FMT_LIB="../fmt/build"`|

#### Examples

Suppose your directory structure is like this:

```
dev/ + berts.cpp/
     |  + berts/
     |  + ggml/
     |
     + fmt/
        + include/
        |  + fmt/
        + build/
           + libfmt.a
```

and if you want to specify the location of `{fmt}`, `make` command will be:

```bash
dev/berts.cpp $ BERTS_DEBUG=1 BERTS_USE_FMTLIB_FMT=1 BERTS_FMTLIB_FMT_INCLUDE="../fmt/include" BERTS_FMTLIB_FMT_LIB="../fmt/build" make
```

or if you want to specify the location of ICU header:

```bash
dev/berts.cpp $ BERTS_DEBUG=1 BERTS_ICU_INCLUDE='C:/Program Files (x86)/Windows Kits/10/Include/10.0.22000.0/um' make
```

## TODO

- tokenizers: BertJapaneseTokenizer, DebertaV2Tokenizer
- model: DeBERTa, DeBERTa-v2
- pooler's act fn (for deberta)
- GPU
- batching (attn mask)
- never_split (any model?)
- load gguf from memory
- load gguf from std::istream
- position_embedding_type: relative_key|relative_key_query
