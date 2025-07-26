"""
Aggressive auto-installer for Danbooru FAISS Lookup dependencies
This tries multiple methods to ensure dependencies get installed
"""

import sys
import subprocess
import os
from pathlib import Path

def detect_install_type():
    """Detect the installation type based on Python executable"""
    python_exec = sys.executable
    
    if "python_embeded" in python_exec or "python_embedded" in python_exec:
        return "portable"
    elif ".venv" in python_exec or "venv" in python_exec:
        return "venv"
    else:
        return "system"

def try_install_package(package_name, pip_args=None):
    """Try to install a single package"""
    if pip_args is None:
        pip_args = []
    
    install_type = detect_install_type()
    
    # Build the command based on install type
    if install_type == "portable":
        cmd = [sys.executable, "-s", "-m", "pip", "install"] + pip_args + [package_name]
    else:
        cmd = [sys.executable, "-m", "pip", "install"] + pip_args + [package_name]
    
    try:
        print(f"[{install_type}] Installing: {package_name}")
        result = subprocess.run(cmd, capture_output=False)
        if result.returncode == 0:
            return True
    except Exception as e:
        print(f"Error: {e}")
    
    # Fallback: try without -s flag
    if install_type == "portable":
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + pip_args + [package_name]
            print(f"[{install_type}] Retrying without -s flag...")
            result = subprocess.run(cmd, capture_output=False)
            if result.returncode == 0:
                return True
        except:
            pass
    
    return False

def main():
    """Install all required packages"""
    install_type = detect_install_type()
    print(f"[Danbooru Lookup] Using {install_type} Python: {sys.executable}")
    
    # Core packages (always needed)
    core_packages = [
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "requests>=2.25.0",
        "tqdm>=4.60.0",
        "faiss-cpu>=1.7.0"
    ]
    
    # Additional packages for full functionality
    additional_packages = [
        "huggingface-hub>=0.16.0",
        "Pillow>=9.0.0",
        "jax>=0.4.0",
        "jaxlib>=0.4.0", 
        "flax>=0.7.0"
    ]
    
    # WD14 support (try GPU version first)
    wd14_packages = [
        ("dghs-imgutils[gpu]>=0.17.0", "dghs-imgutils>=0.17.0")  # (preferred, fallback)
    ]
    
    failed = []
    
    print("[Danbooru Lookup] Installing core dependencies...")
    
    # Install core packages
    for package in core_packages:
        print(f"\nInstalling {package}...")
        if not try_install_package(package):
            # Try without version constraint
            package_name = package.split(">=")[0]
            if not try_install_package(package_name):
                # Try with --user flag
                if not try_install_package(package, ["--user"]):
                    failed.append(package)
    
    # Install additional packages
    print("\n[Danbooru Lookup] Installing additional packages...")
    for package in additional_packages:
        print(f"\nInstalling {package}...")
        if not try_install_package(package):
            package_name = package.split(">=")[0]
            try_install_package(package_name)  # Best effort
    
    # Install WD14 support
    print("\n[Danbooru Lookup] Installing WD14 image support...")
    for gpu_version, cpu_version in wd14_packages:
        print(f"\nTrying GPU version: {gpu_version}")
        if not try_install_package(gpu_version):
            print(f"GPU version failed, trying CPU version: {cpu_version}")
            if not try_install_package(cpu_version):
                print("[WARNING] dghs-imgutils installation failed. Image inputs won't work.")
    
    
    if failed:
        print(f"\n[ERROR] Failed to install core dependencies: {', '.join(failed)}")
        return False
    else:
        print("\n[SUCCESS] Core dependencies installed!")
        print("Some optional features may require additional setup.")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)