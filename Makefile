# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O0
DEBUG_FLAGS := -g -G

# Source directory
SRC_DIR := src

# Binary directory
BIN_DIR := bin

# Debug binary directory
DEBUG_BIN_DIR := $(BIN_DIR)/debug

# Get all .cu files in the source directory
SOURCES := $(wildcard $(SRC_DIR)/*.cu)

# Generate executable names
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))
DEBUG_EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu,$(DEBUG_BIN_DIR)/%,$(SOURCES))

# Default target
all: $(EXECUTABLES)

# Debug target
debug: $(DEBUG_EXECUTABLES)

# Rule to compile each source file into an executable
$(BIN_DIR)/%: $(SRC_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

# Rule to compile each source file into a debug executable
$(DEBUG_BIN_DIR)/%: $(SRC_DIR)/%.cu | $(DEBUG_BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $(DEBUG_FLAGS) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

$(DEBUG_BIN_DIR):
	mkdir -p $(DEBUG_BIN_DIR)

# Clean target
clean:
	rm -rf $(BIN_DIR)

# Run all executables
run: $(EXECUTABLES)
	@for exe in $(EXECUTABLES); do \
		echo "Running $$exe"; \
		$$exe; \
		echo; \
	done

# Rule to run a single executable by name
run_%: $(BIN_DIR)/%
	@echo "Running $<"
	@$<

# Rule to run a single executable in debug mode
run_debug_%: $(DEBUG_BIN_DIR)/%
	@echo "Running $< in debug mode with CUDA-GDB"
	@cuda-gdb $<

.PHONY: all debug clean run