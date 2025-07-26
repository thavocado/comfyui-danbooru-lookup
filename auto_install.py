"""
Aggressive auto-installer for Danbooru FAISS Lookup dependencies
This tries multiple methods to ensure dependencies get installed
"""

import sys
import subprocess
import os
from pathlib import Path

def try_install_package(package_name, pip_args=None):
    """Try to install a single package with multiple methods"""
    if pip_args is None:
        pip_args = []
    
    commands = [
        [sys.executable, "-m", "pip", "install"] + pip_args + [package_name],
        [sys.executable, "-s", "-m", "pip", "install"] + pip_args + [package_name],
        ["pip", "install"] + pip_args + [package_name],
        ["pip3", "install"] + pip_args + [package_name],
    ]
    
    for cmd in commands:
        try:
            print(f"Trying: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode == 0:
                return True
        except:
            continue
    return False

def main():
    """Install all required packages"""
    packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "faiss-cpu>=1.7.0"
    ]
    
    failed = []
    
    print("[Danbooru Lookup] Installing dependencies one by one...")
    
    for package in packages:
        print(f"\nInstalling {package}...")
        if not try_install_package(package):
            # Try without version constraint
            package_name = package.split(">=")[0]
            if not try_install_package(package_name):
                # Try with --user flag
                if not try_install_package(package, ["--user"]):
                    failed.append(package)
    
    if failed:
        print(f"\n[ERROR] Failed to install: {', '.join(failed)}")
        return False
    else:
        print("\n[SUCCESS] All dependencies installed!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)