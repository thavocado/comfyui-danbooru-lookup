#!/usr/bin/env python3
"""
Utility to fix corrupted data files by re-downloading them.
Run this if you encounter FAISS read errors.
"""

import sys
import os
from pathlib import Path
import logging

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from modules.data_loader import DataLoader

def main():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
    
    print("=" * 60)
    print("Danbooru FAISS Lookup - Fix Corrupted Files")
    print("=" * 60)
    print()
    
    data_dir = Path(__file__).parent / "data"
    
    # Check if data directory exists
    if not data_dir.exists():
        print("Data directory not found. Files will be downloaded fresh.")
    else:
        print(f"Data directory: {data_dir}")
        print()
        
        # List existing files
        print("Existing files:")
        for subdir in ["index", "data"]:
            subdir_path = data_dir / subdir
            if subdir_path.exists():
                for file in subdir_path.iterdir():
                    if file.is_file():
                        size = file.stat().st_size
                        print(f"  {file.relative_to(data_dir)}: {size:,} bytes")
        print()
    
    # Ask user to confirm
    response = input("Do you want to re-download all data files? This will delete existing files. (y/N): ")
    if response.lower() != 'y':
        print("Cancelled.")
        return
    
    print()
    print("Re-downloading all data files...")
    
    try:
        # Create data loader
        loader = DataLoader(data_dir)
        
        # Force re-download all files
        loader.ensure_data_downloaded(force_redownload=True)
        
        print()
        print("[SUCCESS] All files have been re-downloaded successfully!")
        print("You can now use the Danbooru FAISS Lookup node in ComfyUI.")
        
    except Exception as e:
        print()
        print(f"[ERROR] Failed to download files: {e}")
        print()
        print("Please check your internet connection and try again.")
        print("If the problem persists, please report it at:")
        print("https://github.com/thavocado/comfyui-danbooru-lookup/issues")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())