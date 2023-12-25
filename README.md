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

### Variables

|Name|Effect|
|---  |---   |
|`BERTS_DEBUG`|Enable debugging|
|`BERTS_ICU_INCLUDE`|Specify include path of [ICU4C](https://unicode-org.github.io/icu/userguide/icu4c/) headers relative to `berts` directory|
|`BERTS_ICU_LIB`|Specify search path of [ICU4C](https://unicode-org.github.io/icu/userguide/icu4c/) libs relative to `berts` directory|
|`BERTS_USE_FMTLIB_FMT`|Use [{fmt} lib](https://github.com/fmtlib/fmt) instead of std::format|
|`BERTS_FMTLIB_FMT_INCLUDE`|Specify include path of the {fmt} header relative to `berts` directory (passed to `-I`)|
|`BERTS_FMTLIB_FMT_LIB`|Specify search path for `libfmt.a` relative to `berts` directory (passed to `-L`)|

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

and you want to specify the location of {fmt}, `make` command will be:

```bash
dev/berts.cpp $ BERTS_DEBUG=1 BERTS_USE_FMTLIB_FMT=1 BERTS_FMTLIB_FMT_INCLUDE="../fmt/include" BERTS_FMTLIB_FMT_LIB="../fmt/build" make
```

## TODO

- pooling
- inference
- zstd vocab
- load gguf from memory
- load gguf from std::istream
- move eps from `berts_context` to `internal::hparams`
