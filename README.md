# berts.cpp

[ggml](https://github.com/ggerganov/ggml) implementation of BERT.

## Description

(WIP)

## How to build

```bash
$ git clone --depth 1 https://github.com/hnmr293/berts.cpp --recurse-submodules --shallow-submodules
$ cd berts.cpp
$ make
$ ./berts/main
```

### Macros

|Name|Effect|
|---  |---   |
|`BERTS_DEBUG`|Enable debugging|
|`BERTS_USE_FMTLIB_FMT`|Use [{fmt} lib](https://github.com/fmtlib/fmt) instead of std::format|
|`BERTS_FMTLIB_FMT_INCLUDE`|Specify include path of {fmt} header (passed to `-I`)|
|`BERTS_FMTLIB_FMT_LIB`|Specify search path for `libfmt.a` (passed to `-L`)|

## TODO

- vocab_size -> 8-bit
- pooling
- inference
- zstd vocab
- load gguf from memory
- load gguf from std::istream
