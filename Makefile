# Compiler
NVCC := nvcc

# Compiler flags
NVCC_FLAGS := -O2

# Source directory
SRC_DIR := src

# Build directory
BUILD_DIR := build

# Get all .cu files in the source directory
SOURCES := $(wildcard $(SRC_DIR)/*.cu)

# Generate executable names
EXECUTABLES := $(patsubst $(SRC_DIR)/%.cu,$(BUILD_DIR)/%,$(SOURCES))

# Default target
all: $(EXECUTABLES)

# Rule to compile each source file into an executable
$(BUILD_DIR)/%: $(SRC_DIR)/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCC_FLAGS) $< -o $@

$(BUILD_DIR):
	mkdir -p $(BUILD_DIR)

# Clean target
clean:
	rm -rf $(BUILD_DIR)

# Run all executables
run: $(EXECUTABLES)
	@for exe in $(EXECUTABLES); do \
		echo "Running $$exe"; \
		$$exe; \
		echo; \
	done

# Rule to run a single executable by name
run_%: $(BUILD_DIR)/%
	@echo "Running $<"
	@$<

.PHONY: all clean run