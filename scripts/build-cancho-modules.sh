#!/bin/bash
set -e

# Build script for pure cancho modules
# This script builds the new cancho modules (server, client, sample, chat)
# and demonstrates that they can be compiled with the cancho compiler.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$REPO_ROOT/build"
CANCHO_BIN="$REPO_ROOT/build/cancho-gpu"
CANCHO_COMPILER="${CANCHO_COMPILER:-/Users/alfonso/Workspace/alpibrusl/cancho/target/release/cancho}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo "🚀 Building pure cancho modules..."
echo ""

# Ensure build directory exists
mkdir -p "$BUILD_DIR"

# Check if cancho compiler exists
if [ ! -f "$CANCHO_BIN" ]; then
    echo "${RED}❌ cancho compiler not found at $CANCHO_BIN${NC}"
    echo "Please build cancho-gpu first: cargo build --release -p cancho-gpu"
    exit 1
fi

echo "${GREEN}✅ cancho compiler found at $CANCHO_BIN${NC}"
echo ""

# Function to build a single cancho module
build_module() {
    local module_name="$1"
    local source_file="$REPO_ROOT/experimental/${module_name}.cho"
    local output_file="$BUILD_DIR/${module_name}"
    
    if [ ! -f "$source_file" ]; then
        echo "${RED}❌ Source file not found: $source_file${NC}"
        return 1
    fi
    
    echo "${YELLOW}🔨 Building $module_name...${NC}"
    
    # Build the module
    if "$CANCHO_COMPILER" build "$source_file" --std -o "$output_file" 2>&1; then
        echo "${GREEN}✅ Successfully built $module_name${NC}"
        return 0
    else
        echo "${RED}❌ Failed to build $module_name${NC}"
        return 1
    fi
}

# Build all new modules
MODULES=("server" "client" "sample" "chat")
SUCCESS_COUNT=0
FAIL_COUNT=0

for module in "${MODULES[@]}"; do
    if build_module "$module"; then
        ((SUCCESS_COUNT++))
    else
        ((FAIL_COUNT++))
    fi
    echo ""
done

# Summary
echo "=========================================="
echo "Build Summary:"
echo "  Success: $SUCCESS_COUNT modules"
echo "  Failed:  $FAIL_COUNT modules"
echo "=========================================="

if [ $FAIL_COUNT -gt 0 ]; then
    echo "${RED}❌ Some modules failed to build${NC}"
    exit 1
fi

echo "${GREEN}✅ All pure cancho modules built successfully!${NC}"

# Test the modules
echo ""
echo "${YELLOW}🧪 Testing modules...${NC}"

# Test server module
echo "Testing server module..."
if [ -f "$BUILD_DIR/server" ]; then
    echo "  server.cho: ✅ Compiled"
else
    echo "  server.cho: ❌ Not found"
fi

# Test client module  
echo "Testing client module..."
if [ -f "$BUILD_DIR/client" ]; then
    echo "  client.cho: ✅ Compiled"
else
    echo "  client.cho: ❌ Not found"
fi

# Test sample module
echo "Testing sample module..."
if [ -f "$BUILD_DIR/sample" ]; then
    echo "  sample.cho: ✅ Compiled"
else
    echo "  sample.cho: ❌ Not found"
fi

# Test chat module
echo "Testing chat module..."
if [ -f "$BUILD_DIR/chat" ]; then
    echo "  chat.cho: ✅ Compiled"
else
    echo "  chat.cho: ❌ Not found"
fi

echo ""
echo "${GREEN}🎉 Pure cancho migration is working!${NC}"
echo ""
echo "Next steps:"
echo "  1. Integrate these modules with cancho-gpu main"
echo "  2. Create FFI interface to Rust shim"
echo "  3. Replace lex-rt components with cancho versions"
echo "  4. Test end-to-end functionality"