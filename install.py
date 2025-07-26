import os
import sys
import subprocess
import threading
import locale
import traceback
from pathlib import Path

# Add current directory to path for portable installations
if sys.argv[0] == 'install.py':
    sys.path.append('.')

def handle_stream(stream, is_stdout):
    """Handle subprocess output streams"""
    try:
        stream.reconfigure(encoding=locale.getpreferredencoding(), errors='replace')
    except:
        pass
    
    for msg in stream:
        if is_stdout:
            print(msg, end="", file=sys.stdout)
        else:
            print(msg, end="", file=sys.stderr)

def run_command(cmd, cwd=None):
    """Run a command and handle output"""
    print(f"[Danbooru Lookup] Running: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd, 
        cwd=cwd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True, 
        bufsize=1
    )
    
    stdout_thread = threading.Thread(target=handle_stream, args=(process.stdout, True))
    stderr_thread = threading.Thread(target=handle_stream, args=(process.stderr, False))
    
    stdout_thread.start()
    stderr_thread.start()
    
    stdout_thread.join()
    stderr_thread.join()
    
    return process.wait()

def detect_python_executable():
    """Detect the appropriate Python executable"""
    # Use the Python that's currently running this script
    # This ensures we use the same Python that ComfyUI is using
    python_exec = sys.executable
    
    # Determine install type based on the executable path
    if "python_embeded" in python_exec or "python_embedded" in python_exec:
        install_type = "portable"
    elif ".venv" in python_exec or "venv" in python_exec:
        install_type = "venv"
    else:
        install_type = "system"
    
    return python_exec, install_type

def install_requirements():
    """Install requirements using the appropriate Python executable"""
    python_exec, install_type = detect_python_executable()
    requirements_file = Path(__file__).parent / "requirements.txt"
    
    print(f"### Danbooru FAISS Lookup: Installing dependencies")
    print(f"Using {install_type} Python: {python_exec}")
    
    # Build pip install command
    # Use -s flag for portable to ignore user site-packages
    if install_type == "portable":
        cmd = [python_exec, "-s", "-m", "pip", "install", "-r", str(requirements_file)]
    else:
        cmd = [python_exec, "-m", "pip", "install", "-r", str(requirements_file)]
    
    # Run installation
    result = run_command(cmd, cwd=str(Path(__file__).parent))
    
    # If dghs-imgutils[gpu] failed, try without GPU
    if result != 0:
        print("\n[WARNING] Some packages failed. Trying fallback installation...")
        
        # Install core packages first
        core_file = Path(__file__).parent / "requirements-core.txt"
        if core_file.exists():
            print("\nInstalling core packages...")
            if install_type == "portable":
                cmd = [python_exec, "-s", "-m", "pip", "install", "-r", str(core_file)]
            else:
                cmd = [python_exec, "-m", "pip", "install", "-r", str(core_file)]
            run_command(cmd, cwd=str(Path(__file__).parent))
        
        # Try dghs-imgutils without GPU
        print("\nTrying dghs-imgutils without GPU support...")
        if install_type == "portable":
            cmd = [python_exec, "-s", "-m", "pip", "install", "dghs-imgutils>=0.17.0"]
        else:
            cmd = [python_exec, "-m", "pip", "install", "dghs-imgutils>=0.17.0"]
        run_command(cmd, cwd=str(Path(__file__).parent))
        
        # Install other recommended packages
        for package in ["huggingface-hub>=0.16.0", "Pillow>=9.0.0"]:
            if install_type == "portable":
                cmd = [python_exec, "-s", "-m", "pip", "install", package]
            else:
                cmd = [python_exec, "-m", "pip", "install", package]
            run_command(cmd, cwd=str(Path(__file__).parent))
    
    print("\n[INFO] Installation complete.")
    
    return True

def main():
    """Main installation function"""
    print("=" * 60)
    print("### ComfyUI Danbooru FAISS Lookup - Dependency Installer ###")
    print("=" * 60)
    
    try:
        # Check if requirements are already satisfied
        try:
            import faiss
            import pandas
            import tqdm
            print("Dependencies appear to be already installed.")
            print("Checking for updates...")
        except ImportError:
            print("Missing dependencies detected.")
        
        # Install requirements
        success = install_requirements()
        
        if success:
            print("\n[COMPLETE] Installation finished successfully!")
            print("You can now restart ComfyUI and use the Danbooru FAISS Lookup node.")
        else:
            print("\n[WARNING] Installation completed with errors.")
            print("Some dependencies may need to be installed manually.")
        
    except Exception as e:
        print(f"\n[ERROR] Installation failed with error: {e}")
        traceback.print_exc()
        print("\nPlease install dependencies manually:")
        print(f"  pip install -r {Path(__file__).parent / 'requirements.txt'}")

if __name__ == "__main__":
    main()