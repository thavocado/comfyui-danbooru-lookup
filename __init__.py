"""
@author: ComfyUI Danbooru Lookup
@title: Danbooru FAISS Lookup
@nickname: Danbooru Lookup
@description: ComfyUI node that performs FAISS cosine similarity lookup on Danbooru embeddings using CLIP conditioning inputs.
@version: 1.0.0
"""

import sys
import os
import traceback
from pathlib import Path

# Track if dependencies are available
DEPENDENCIES_INSTALLED = True
MISSING_DEPENDENCIES = []

# Check for dependencies without importing them yet
try:
    import faiss
    import pandas
    import numpy
    import tqdm
except ImportError as e:
    DEPENDENCIES_INSTALLED = False
    print(f"[Danbooru Lookup] Missing dependencies detected: {e}")
    print("[Danbooru Lookup] Attempting to install dependencies...")
    
    # Try to install dependencies directly
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            import subprocess
            
            # Try different pip installation methods
            pip_commands = [
                [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)],
                [sys.executable, "-s", "-m", "pip", "install", "-r", str(requirements_file)],  # For portable Python
                ["pip", "install", "-r", str(requirements_file)],
                ["pip3", "install", "-r", str(requirements_file)],
            ]
            
            install_success = False
            for cmd in pip_commands:
                try:
                    print(f"[Danbooru Lookup] Trying: {' '.join(cmd)}")
                    # Don't capture output so we can see what's happening
                    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
                    if result.returncode == 0:
                        print("[Danbooru Lookup] Dependencies installed successfully!")
                        install_success = True
                        break
                except Exception as cmd_error:
                    continue
            
            if not install_success:
                print("[Danbooru Lookup] Auto-installation failed. Trying auto_install.py...")
                # Try to run auto_install.py as fallback
                auto_install_script = Path(__file__).parent / "auto_install.py"
                if auto_install_script.exists():
                    try:
                        result = subprocess.run([sys.executable, str(auto_install_script)], cwd=str(Path(__file__).parent))
                        if result.returncode == 0:
                            install_success = True
                    except:
                        pass
                
                if not install_success:
                    # Last resort: try install.py
                    install_script = Path(__file__).parent / "install.py"
                    if install_script.exists():
                        try:
                            subprocess.run([sys.executable, str(install_script)], cwd=str(Path(__file__).parent))
                        except:
                            pass
                        
        except Exception as install_error:
            print(f"[Danbooru Lookup] Installation error: {install_error}")
    
    print("[Danbooru Lookup] Please install dependencies manually:")
    print(f"  1. Navigate to: {Path(__file__).parent}")
    print(f"  2. Run: python install.py")
    print(f"  OR")
    print(f"  Run: pip install -r {Path(__file__).parent / 'requirements.txt'}")

# Create a stub node if dependencies are missing
if not DEPENDENCIES_INSTALLED:
    class DanbooruFAISSLookupStub:
        """Stub node shown when dependencies are not installed"""
        
        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "positive_conditioning": ("CONDITIONING",),
                    "negative_conditioning": ("CONDITIONING",),
                }
            }
        
        RETURN_TYPES = ("STRING",)
        RETURN_NAMES = ("danbooru_id",)
        FUNCTION = "lookup"
        CATEGORY = "conditioning/danbooru"
        
        def lookup(self, positive_conditioning, negative_conditioning, **kwargs):
            error_msg = (
                "ERROR: Danbooru FAISS Lookup dependencies not installed!\n"
                f"Please run: {Path(__file__).parent / 'install.py'}\n"
                f"Or install manually: pip install -r {Path(__file__).parent / 'requirements.txt'}"
            )
            print(f"[Danbooru Lookup] {error_msg}")
            return ("ERROR: Dependencies not installed",)
    
    NODE_CLASS_MAPPINGS = {
        "DanbooruFAISSLookup": DanbooruFAISSLookupStub
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DanbooruFAISSLookup": "Danbooru FAISS Lookup (INSTALL REQUIRED)"
    }
else:
    # Dependencies are installed, load the real node
    try:
        from .modules.danbooru_lookup import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
    except Exception as e:
        print(f"[ERROR] Failed to load Danbooru FAISS Lookup node: {e}")
        traceback.print_exc()
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]