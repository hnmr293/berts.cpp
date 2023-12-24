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

## TODO

- pooling
- inference
- zstd vocab
- load gguf from memory
- load gguf from std::istream
- move eps from `berts_context` to `internal::hparams`
