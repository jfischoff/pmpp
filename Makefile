# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O2

# Source directory
SRC_DIR := src

# Binary directory
BIN_DIR := bin

# Get all .cu files in the source directory
SOURCES := $(wildcard $(SRC_DIR)/*.cu)

# Generate executable names
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu,$(BIN_DIR)/%,$(SOURCES))

# Default target
all: $(EXECUTABLES)

# Rule to compile each source file into an executable
$(BIN_DIR)/%: $(SRC_DIR)/%.cu | $(BIN_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BIN_DIR):
	mkdir -p $(BIN_DIR)

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

.PHONY: all clean run