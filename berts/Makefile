# detect WIN32 and MinGW
ifndef UNAME_S
UNAME_S := $(shell uname -s)
endif

ifneq ($(findstring _NT,$(UNAME_S)),)
_WIN32 := 1
# use *.lib
WIN_LIB := 1
endif

ifneq '' '$(findstring mingw,$(shell $(CC) -dumpmachine))'
_MINGW := 1
# use lib*.a
WIN_LIB := 0
endif

#
# lib prefix and suffix
#
ifneq ($(WIN_LIB),1)
LIB_PRE := lib
SO_EXT := .a
DSO_EXT := .so
else
LIB_PRE :=
SO_EXT := .lib
DSO_EXT := .dll
endif

ifneq ($(_WIN32),1)
EXE_EXT := 
else
EXE_EXT := .exe
endif

BUILD_TARGET = \
	$(STATIC_LIB) \
	$(DYNAMIC_LIB)

EXAMPLES = \
	quant

TESTS = \
	test_tokenizer \
	test_unicode \
	test_trie \
	test_bert \
	test_bpe

BUILD_TARGET += $(addsuffix $(EXE_EXT),$(EXAMPLES))
BUILD_TARGET += $(addsuffix _d$(EXE_EXT),$(EXAMPLES))

BUILD_TARGET += $(addsuffix $(EXE_EXT),$(TESTS))
BUILD_TARGET += $(addsuffix _d$(EXE_EXT),$(TESTS))

default: $(BUILD_TARGET)

MK_CPPFLAGS = -I$(shell pwd)/.. -I$(shell pwd)/../ggml/include
MK_CFLAGS = -std=c11 -fPIC
MK_CXXFLAGS = -std=c++23 -fPIC

# avoid w64devkit bug
MK_CPPFLAGS += -fno-rtti

ifneq ($(_MINGW),1)
# do nothing
else
# The stack is only 16-byte aligned on Windows, so don't let gcc emit aligned moves.
# https://gcc.gnu.org/bugzilla/show_bug.cgi?id=54412
# https://github.com/ggerganov/llama.cpp/issues/2922
MK_CFLAGS   += -Xassembler -muse-unaligned-vector-move
MK_CXXFLAGS += -Xassembler -muse-unaligned-vector-move
endif

#
# external libs
#
ifdef BERTS_USE_FMTLIB_FMT
MK_CPPFLAGS += -DBERTS_USE_FMTLIB_FMT
MK_LDFLAGS += -lfmt
ifdef BERTS_FMTLIB_FMT_INCLUDE
MK_CXXFLAGS += -I'$(BERTS_FMTLIB_FMT_INCLUDE)'
endif
ifdef BERTS_FMTLIB_FMT_LIB
MK_LDFLAGS += -L'$(BERTS_FMTLIB_FMT_LIB)'
endif
endif

MK_LDFLAGS += -licu

ifdef BERTS_ICU_INCLUDE
MK_CPPFLAGS += -I'$(BERTS_ICU_INCLUDE)'
endif

ifdef BERTS_ICU_LIB
MK_LDFLAGS += -L'$(BERTS_ICU_LIB)'
endif

#
# optimization
#
ifdef BERTS_DEBUG
MK_CPPFLAGS += -DDEBUG -DBERTS_DEBUG -DGGML_PERF
MK_CFLAGS += -O0 -g
MK_CXXFLAGS += -O0 -g
MK_LDFLAGS += -g
else
MK_CPPFLAGS += -DNDEBUG
MK_CFLAGS += -Ofast -g -march=native -mtune=native
MK_CXXFLAGS += -Ofast -g -march=native -mtune=native
#NVCCFLAGS += -O3
endif

#
# warnings
#
WARN_FLAGS = -Wall -Wextra -Wpedantic -Wcast-qual -Wno-unused-function
MK_CFLAGS += $(WARN_FLAGS) -Wshadow -Wstrict-prototypes -Wpointer-arith \
		     -Wmissing-prototypes -Werror=implicit-int -Werror=implicit-function-declaration
MK_CXXFLAGS += $(WARN_FLAGS) -Wmissing-declarations -Wmissing-noreturn

#
# lib targets
#
STATIC_LIB := ../lib/$(LIB_PRE)berts$(SO_EXT)
DYNAMIC_LIB := ../lib/$(LIB_PRE)berts$(DSO_EXT)
STAITC_LIB_GGML := ../lib/$(LIB_PRE)ggml$(SO_EXT)
STAITC_LIB_GGML_D := ../lib/$(LIB_PRE)ggml_d$(SO_EXT)

#
# cuBLAS
#
ifdef BERTS_CUBLAS
# todo
endif

#
# resulted flags
#
override CFLAGS   := $(MK_CPPFLAGS) $(CPPFLAGS) $(MK_CFLAGS) $(CFLAGS)
override CXXFLAGS := $(MK_CPPFLAGS) $(CPPFLAGS) $(MK_CXXFLAGS) $(CXXFLAGS)
override LDFLAGS  := $(MK_LDFLAGS) $(LDFLAGS)

$(info I berts.cpp build info: )
$(info I _WIN32:   $(_WIN32))
$(info I _MINGW:   $(_MINGW))
$(info I WIN_LIB:  $(WIN_LIB))
$(info I CFLAGS:   $(CFLAGS))
$(info I CXXFLAGS: $(CXXFLAGS))
$(info I LDFLAGS:  $(LDFLAGS))
$(info I CC:       $(shell $(CC) --version | head -n 1))
$(info I CXX:      $(shell $(CXX) --version | head -n 1))
$(info )

#
# build libs
#

COMMON_HEADERS := berts.h berts.hpp

berts.o: berts.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

gguf.o: models/gguf.cpp models/gguf.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

bert.o: models/bert.cpp models/bert.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

quantize.o: models/quantize.cpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

internal.o: models/internal.cpp models/internal.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

trie.o: models/trie.cpp models/trie.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

unicode.o: models/unicode.cpp models/unicode.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

log.o: models/log.cpp models/log.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

utils.o: models/utils.cpp models/utils.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.lib *.so *.dll *.exe
	rm -f ../lib/libberts.a ../lib/berts.a ../lib/libberts.lib ../lib/berts.lib ../lib/libberts.so ../lib/berts.so ../lib/libberts.dll ../lib/berts.dll

OBJS += utils.o berts.o gguf.o bert.o quantize.o internal.o trie.o unicode.o log.o 

$(STATIC_LIB): $(OBJS)
	ar rcs $@ $^

$(DYNAMIC_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

#
# examples
#

quant$(EXE_EXT):   examples/quant.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

quant_d$(EXE_EXT): examples/quant.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)

#
# tests
#

test_tokenizer$(EXE_EXT):   tests/test_tokenizer.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

test_tokenizer_d$(EXE_EXT): tests/test_tokenizer.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)

test_unicode$(EXE_EXT):     tests/test_unicode.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

test_unicode_d$(EXE_EXT):   tests/test_unicode.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)

test_trie$(EXE_EXT):        tests/test_trie.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

test_trie_d$(EXE_EXT):      tests/test_trie.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)

test_bert$(EXE_EXT):        tests/test_bert.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

test_bert_d$(EXE_EXT):      tests/test_bert.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)

test_bpe$(EXE_EXT):         tests/test_bpe.cpp  $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

test_bpe_d$(EXE_EXT):       tests/test_bpe.cpp  $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)
