# Compiler settings
CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra

# Include headers from workspace and whisper.cpp submodules
INCLUDES := -I. \
            -I./whisper.cpp \
            -I./whisper.cpp/include \
            -I./whisper.cpp/ggml/include

# Dynamic path resolution to find where libwhisper.so actually lives
WHISPER_BUILD_DIR := $(shell find ./whisper.cpp/build -type f -name "libwhisper.so*" -exec dirname {} \; | head -n 1)

# Link options:
# -L passes the directory to search for libwhisper.so
# -Wl,-rpath embeds the runtime search path into your executable so it runs seamlessly
LDFLAGS  := -L$(WHISPER_BUILD_DIR) -Wl,-rpath,$(WHISPER_BUILD_DIR) -lwhisper -lpthread

# Executable name
TARGET   := transcribe_pipeline

# Source files
SRCS     := main.cpp
OBJ      := main.o

.PHONY: all clean info

all: info $(TARGET)

info:
	@echo "[INFO] Found libwhisper at: $(WHISPER_BUILD_DIR)"

# Compile main.cpp
$(OBJ): $(SRCS)
	@echo "[CXX]  $<"
	@$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

# Link using the pre-built shared library
$(TARGET): $(OBJ)
	@echo "[LINK] $@"
	@$(CXX) $(CXXFLAGS) $(OBJ) $(LDFLAGS) -o $@

# Clean target
clean:
	@echo "[CLEAN] Removing local build artifacts..."
	@rm -f $(OBJ) $(TARGET) temp_normalized_16k.wav