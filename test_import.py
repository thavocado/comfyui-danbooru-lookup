#!/usr/bin/env python3
"""Test script to verify JAX import fixes."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing JAX import fixes...")

# Test 1: Import the module
print("\n1. Testing module import...")
try:
    from modules.model_manager import ModelManager
    print("✓ ModelManager imported successfully")
except Exception as e:
    print(f"✗ ModelManager import failed: {e}")

# Test 2: Check dependencies
print("\n2. Testing dependency check...")
try:
    mm = ModelManager()
    deps = mm.check_dependencies()
    print(f"✓ Dependencies checked: {deps}")
except Exception as e:
    print(f"✗ Dependency check failed: {e}")

# Test 3: Create advanced node
print("\n3. Testing advanced node creation...")
try:
    from modules.danbooru_lookup_advanced import DanbooruFAISSLookupAdvanced
    node = DanbooruFAISSLookupAdvanced()
    print("✓ Advanced node created successfully")
except Exception as e:
    print(f"✗ Advanced node creation failed: {e}")

# Test 4: Create another instance
print("\n4. Testing second instance creation...")
try:
    node2 = DanbooruFAISSLookupAdvanced()
    print("✓ Second instance created successfully")
except Exception as e:
    print(f"✗ Second instance creation failed: {e}")

print("\nAll tests completed!")