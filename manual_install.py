#!/usr/bin/env python3
"""
Manual installation helper for Danbooru FAISS Lookup
Run this directly to see detailed error messages and debug installation issues.
"""

import sys
import subprocess
import os
from pathlib import Path

def run_command(cmd, description):
    """Run a command and show detailed output."""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("\nSTDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"\n✓ SUCCESS: {description}")
        else:
            print(f"\n✗ FAILED: {description} (exit code: {result.returncode})")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print(f"\n✗ TIMEOUT: {description} took too long")
        return False
    except Exception as e:
        print(f"\n✗ ERROR: {e}")
        return False

def clear_python_cache():
    """Clear Python cache files that might be causing import issues."""
    print("\nClearing Python cache...")
    script_dir = Path(__file__).parent
    cache_dirs = list(script_dir.glob("**/__pycache__"))
    
    if cache_dirs:
        for cache_dir in cache_dirs:
            try:
                import shutil
                shutil.rmtree(cache_dir)
                print(f"  Removed: {cache_dir}")
            except Exception as e:
                print(f"  Failed to remove {cache_dir}: {e}")
    else:
        print("  No cache directories found.")

def main():
    print("Danbooru FAISS Lookup - Manual Installation Helper")
    print(f"Python: {sys.executable}")
    print(f"Version: {sys.version}")
    print(f"Working directory: {os.getcwd()}")
    
    # Detect portable installation
    if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
        print("\n*** PORTABLE COMFYUI DETECTED ***")
        print("This can sometimes cause installation issues.")
        print("If installation fails repeatedly, try:")
        print("  1. Clear Python cache (this script will do it)")
        print("  2. Install packages one by one")
        print("  3. Restart ComfyUI after installation")
    
    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    print(f"Changed to: {os.getcwd()}")
    
    # Check if requirements.txt exists
    req_file = script_dir / "requirements.txt"
    if not req_file.exists():
        print(f"\nERROR: requirements.txt not found at {req_file}")
        return 1
    
    print(f"\nFound requirements.txt at: {req_file}")
    
    # Step 0: Clear Python cache
    print("\n" + "="*60)
    print("STEP 0: Clearing Python cache...")
    clear_python_cache()
    
    # Step 1: Try updating pip first
    print("\n" + "="*60)
    print("STEP 1: Updating pip...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], "Update pip")
    
    # Step 2: Try installing all requirements
    print("\n" + "="*60)
    print("STEP 2: Installing all requirements...")
    success = run_command([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"], "Install all requirements")
    
    if not success:
        print("\n" + "="*60)
        print("STEP 3: Trying individual packages...")
        
        # Read requirements
        with open(req_file, 'r') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        
        failed = []
        for req in requirements:
            if not run_command([sys.executable, "-m", "pip", "install", req], f"Install {req}"):
                failed.append(req)
        
        if failed:
            print("\n" + "="*60)
            print("FAILED PACKAGES:")
            for pkg in failed:
                print(f"  - {pkg}")
            
            # Special handling for common issues
            print("\n" + "="*60)
            print("TRYING FALLBACKS...")
            
            for pkg in failed:
                if "dghs-imgutils[gpu]" in pkg:
                    print("\nTrying dghs-imgutils without GPU support...")
                    run_command([sys.executable, "-m", "pip", "install", "dghs-imgutils"], "Install dghs-imgutils (CPU)")
                elif "jax" in pkg.lower():
                    print("\nTrying JAX with proper installation order...")
                    # For portable installations, sometimes we need --no-cache-dir
                    cache_flag = ["--no-cache-dir"] if "python_embeded" in sys.executable else []
                    run_command([sys.executable, "-m", "pip", "install"] + cache_flag + ["jaxlib"], "Install jaxlib first")
                    run_command([sys.executable, "-m", "pip", "install"] + cache_flag + ["jax"], "Install jax")
                    run_command([sys.executable, "-m", "pip", "install"] + cache_flag + ["flax"], "Install flax")
                elif "flax" in pkg.lower():
                    # Skip if already handled with jax
                    continue
    
    # Step 4: Verify installation
    print("\n" + "="*60)
    print("STEP 4: Verifying installation...")
    
    packages_to_check = [
        ("faiss", "import faiss"),
        ("pandas", "import pandas"),
        ("numpy", "import numpy"),
        ("tqdm", "import tqdm"),
        ("PIL", "import PIL"),
        ("imgutils", "from imgutils.tagging import wd14"),
        ("jax", "import jax"),
        ("flax", "import flax"),
    ]
    
    all_good = True
    for name, import_cmd in packages_to_check:
        try:
            exec(import_cmd)
            print(f"✓ {name} is working")
        except ImportError as e:
            print(f"✗ {name} failed: {e}")
            all_good = False
    
    print("\n" + "="*60)
    if all_good:
        print("SUCCESS! All packages are installed correctly.")
        print("Please restart ComfyUI to use the Danbooru FAISS Lookup node.")
    else:
        print("Some packages failed to install.")
        print("\nCommon solutions:")
        print("1. Check your internet connection")
        print("2. Try using a VPN if you're in a region with restricted access")
        print("3. On Windows: Install Visual C++ Build Tools")
        print("4. Try running: python -m pip install --upgrade pip setuptools wheel")
        print("5. Check for conflicting packages in your environment")
    
    return 0 if all_good else 1

if __name__ == "__main__":
    sys.exit(main())