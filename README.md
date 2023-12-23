# berts.cpp

[ggml](https://github.com/ggerganov/ggml) implementation of BERT.

## Description

(WIP)

## How to build

```bash
$ git clone --depth 1 https://github.com/hnmr293/berts.cpp --recursive-submodules --shallow-submodules
$ cd berts.cpp

# make ggml
$ cd ggml
$ mkdir build && cd build
$ cmake ..
$ make
$ cd ../

# make berts.cpp
$ make -j
# for debugging
$ BERTS_DEBUG=1 make -j
```

## TODO

- quantize model
- zstd vocab
- load gguf from memory
- load gguf from std::istream
