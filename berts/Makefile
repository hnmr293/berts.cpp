BUILD_TARGET = \
	$(STATIC_LIB) \
	$(DYNAMIC_LIB)

EXAMPLES = \
	quant

BUILD_TARGET += $(EXAMPLES) $(addsuffix _d,$(EXAMPLES))

default: $(BUILD_TARGET)

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

MK_CPPFLAGS = -I$(shell pwd)/.. -I$(shell pwd)/../ggml/include
MK_CFLAGS = -std=c11 -fPIC
MK_CXXFLAGS = -std=c++23 -fPIC

# avoid w64devkit bug
MK_CPPFLAGS += -fno-rtti

# add external library info
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
MK_CPPFLAGS += -DDEBUG -DBERTS_DEBUG
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
# lib prefix and suffix
#
ifneq ($(WIN_LIB),1)
LIB_PRE := lib
endif

ifneq ($(WIN_LIB),1)
SO_EXT := .a
else
SO_EXT := .lib
endif

ifneq ($(WIN_LIB),1)
DSO_EXT := .so
else
DSO_EXT := .dll
endif

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

tokenizer.o: models/tokenizer.cpp models/tokenizer.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

trie.o: models/trie.cpp models/trie.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

unicode.o: models/unicode.cpp models/unicode.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

internal.o: models/internal.cpp models/internal.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

log.o: common/log.cpp common/log.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.lib *.so *.dll *.exe
	rm -f ../lib/libberts.a ../lib/berts.a ../lib/libberts.lib ../lib/berts.lib ../lib/libberts.so ../lib/berts.so ../lib/libberts.dll ../lib/berts.dll

OBJS += berts.o gguf.o bert.o quantize.o tokenizer.o trie.o unicode.o internal.o log.o 

$(STATIC_LIB): $(OBJS)
	ar rcs $@ $^

$(DYNAMIC_LIB): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

#
# examples
#

quant: examples/quant.cpp   $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML)   -o $@ $(LDFLAGS)

quant_d: examples/quant.cpp $(COMMON_HEADERS) $(STATIC_LIB)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) $(STAITC_LIB_GGML_D) -o $@ $(LDFLAGS)
