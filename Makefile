define newline


endef

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

#
# ggml build config
#
STAITC_LIB_GGML := $(LIB_PRE)ggml$(SO_EXT)
STAITC_LIB_GGML_D := $(LIB_PRE)ggml_d$(SO_EXT)

GGML_OBJ_NAMES := ggml ggml-alloc ggml-backend ggml-quants
ifneq ($(_WIN32),1)
GGML_OBJ_NAMES := $(addsuffix .c.o,$(GGML_OBJ_NAMES))
else
GGML_OBJ_NAMES := $(addsuffix .c.obj,$(GGML_OBJ_NAMES))
endif

GGML_BUILD_DIR := ggml/build
GGML_BUILD_DIR_D := ggml/build-debug

GGML_OBJ_DIR := $(GGML_BUILD_DIR)/src/CMakeFiles/ggml.dir
GGML_OBJ_DIR_D := $(GGML_BUILD_DIR_D)/src/CMakeFiles/ggml.dir
GGML_LIB_DIR := lib
GGML_LIB_DIR_D := lib

GGML_OBJ := $(addprefix $(GGML_OBJ_DIR)/,$(GGML_OBJ_NAMES))
GGML_OBJ_D := $(addprefix $(GGML_OBJ_DIR_D)/,$(GGML_OBJ_NAMES))

ifneq ($(_WIN32),1)
GGML_CMAKE := cmake ..
else
GGML_CMAKE := cmake .. -G "MinGW Makefiles"
endif

GGML_CMAKE += -DGGML_BUILD_TESTS=OFF -DGGML_BUILD_EXAMPLES=OFF
GGML_CMAKE_D = $(GGML_CMAKE) -DCMAKE_BUILD_TYPE=Debug



all: ggml ggml_d berts

berts: ggml
	make -C berts

ggml: $(GGML_LIB_DIR)/$(STAITC_LIB_GGML)

ggml_d: $(GGML_LIB_DIR_D)/$(STAITC_LIB_GGML_D)

$(GGML_LIB_DIR)/$(STAITC_LIB_GGML): $(GGML_OBJ)
	ar rcs lib/$(STAITC_LIB_GGML) $^

$(GGML_LIB_DIR_D)/$(STAITC_LIB_GGML_D): $(GGML_OBJ_D)
	ar rcs lib/$(STAITC_LIB_GGML_D) $^

$(GGML_OBJ):
	mkdir -p $(GGML_BUILD_DIR) && \
	cd $(GGML_BUILD_DIR) && \
	$(GGML_CMAKE) && \
	make -j

$(GGML_OBJ_D):
	mkdir -p $(GGML_BUILD_DIR_D) && \
	cd $(GGML_BUILD_DIR_D) && \
	$(GGML_CMAKE_D) && \
	make -j

ggml_error:
	$(warning Please build ggml first, then build berts.cpp again.)
	$(warning *** === How to build ggml === )
	$(warning *** $$ mkdir -p ggml/build && cd ggml/build)
	$(warning *** $$ cmake ..)
	$(warning *** (for *nix) or)
	$(warning *** $$ cmake .. -G "MinGW Makefiles")
	$(warning *** (for MinGW))
	$(warning *** $$ make -j)
	$(warning *** You can pass -DCMAKE_BUILD_TYPE=Debug to cmake for debugging)
	$(error )

clean:
	make clean -C berts
	rm -rf $(GGML_BUILD_DIR)
	rm -rf $(GGML_BUILD_DIR_D)
	rm -f lib/*.a lib/*.lib lib/*.so lib/*.dll
