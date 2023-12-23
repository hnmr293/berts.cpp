BUILD_TARGET = \
	$(STATIC_LIB_NAME) \
	$(DYNAMIC_LIB_NAME) \
	main

default: $(BUILD_TARGET)

MK_CPPFLAGS = -I$(shell pwd)/.. -I$(shell pwd)/../ggml/include
MK_CFLAGS = -std=c11 -fPIC
MK_CXXFLAGS = -std=c++23 -fPIC

#
# optimization
#
MK_CFLAGS += -Ofast -march=native -mtune=native
MK_CXXFLAGS += -Ofast -march=native -mtune=native
#NVCCFLAGS += -O3

ifdef BERTS_DEBUG
MK_CPPFLAGS += -DDEBUG -DBERTS_DEBUG
MK_CFLAGS += -O0 -g
MK_CXXFLAGS += -O0 -g
MK_LDFLAGS += -g
else
MK_CPPFLAGS += -DNDEBUG
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
ifneq ($(findstring _NT,$(UNAME_S)),)
_WIN32 := 1
endif

ifneq ($(_WIN32),1)
LIB_PRE := lib
endif

ifneq ($(_WIN32),1)
DSO_EXT := .so
else
DSO_EXT := .dll
endif

STATIC_LIB_NAME := $(LIB_PRE)berts.lib
DYNAMIC_LIB_NAME := $(LIB_PRE)berts$(DSO_EXT)

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

internal.o: models/internal.cpp models/internal.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

log.o: models/log.cpp models/log.hpp $(COMMON_HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

clean:
	rm -rf *.o *.lib *.so *.dll *.exe

OBJS += berts.o gguf.o bert.o internal.o log.o 

$(STATIC_LIB_NAME): $(OBJS)
	ar rcs $@ $^

$(DYNAMIC_LIB_NAME): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared -fPIC -o $@ $^ $(LDFLAGS)

#
# examples
#

main: examples/main.cpp $(COMMON_HEADERS) $(STATIC_LIB_NAME)
	$(CXX) $(CXXFLAGS) $(filter-out %.h %.hpp,$^) -o $@ $(LDFLAGS)
