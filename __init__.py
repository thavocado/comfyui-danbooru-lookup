"""
@author: ComfyUI Danbooru Lookup
@title: Danbooru FAISS Lookup
@nickname: Danbooru Lookup
@description: ComfyUI node that performs FAISS cosine similarity lookup on Danbooru embeddings using CLIP conditioning inputs.
@version: 1.0.1
"""

# Version marker for debugging
print("[Danbooru Lookup] Loading v1.0.1 with improved installation...")

import sys
import os
import traceback
from pathlib import Path
import subprocess

# Fix for OpenMP conflict between different libraries (FAISS, NumPy, etc.)
# This must be set before importing any libraries that use OpenMP
if os.name == 'nt':  # Windows only
    # Check if not already set to avoid overriding user preferences
    if 'KMP_DUPLICATE_LIB_OK' not in os.environ:
        os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
        print("[Danbooru Lookup] Set KMP_DUPLICATE_LIB_OK=TRUE to handle OpenMP conflicts")

# Track if dependencies are available
DEPENDENCIES_INSTALLED = True
MISSING_DEPENDENCIES = []

# Check for ALL dependencies without importing them yet
missing_deps = []
try:
    import faiss
except ImportError:
    missing_deps.append("faiss-cpu")
    
try:
    import pandas
except ImportError:
    missing_deps.append("pandas")
    
try:
    import numpy
except ImportError:
    missing_deps.append("numpy")
    
try:
    import tqdm
except ImportError:
    missing_deps.append("tqdm")
    
try:
    from imgutils.tagging import wd14
except ImportError:
    missing_deps.append("dghs-imgutils")
    
try:
    import PIL
except ImportError:
    missing_deps.append("Pillow")
    
try:
    import jax
    import flax
except ImportError:
    missing_deps.append("jax/flax")

if missing_deps:
    DEPENDENCIES_INSTALLED = False
    print(f"[Danbooru Lookup] Missing dependencies detected: {', '.join(missing_deps)}")
    print("[Danbooru Lookup] Attempting to install dependencies...")
    
    # Try to install dependencies directly
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            print(f"[Danbooru Lookup] Installing dependencies...")
            
            # Simple approach - just use sys.executable like other nodes
            cmd = [sys.executable, "-m", "pip", "install", "-r", str(requirements_file)]
            
            install_success = False
            try:
                print(f"[Danbooru Lookup] Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    print("[Danbooru Lookup] Dependencies installed successfully!")
                    install_success = True
                else:
                    print(f"[Danbooru Lookup] Installation failed with code {result.returncode}")
                    if result.stderr and "Access is denied" in result.stderr:
                        print("[Danbooru Lookup] Access denied error - some packages are in use")
                        print("[Danbooru Lookup] Trying to install missing packages only...")
                        
                        # Try installing only the missing packages
                        for dep in missing_deps:
                            if dep == "dghs-imgutils":
                                pkg = "dghs-imgutils>=0.17.0"
                            elif dep == "jax/flax":
                                # Skip JAX for now if there's a lock issue
                                continue
                            else:
                                pkg = dep
                            
                            try:
                                cmd = [sys.executable, "-m", "pip", "install", pkg]
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                if result.returncode == 0:
                                    print(f"[Danbooru Lookup] Installed {pkg}")
                                else:
                                    print(f"[Danbooru Lookup] Failed to install {pkg}")
                                    if result.stderr:
                                        # Show first few lines of error
                                        error_lines = result.stderr.strip().split('\n')[:3]
                                        for line in error_lines:
                                            print(f"[Danbooru Lookup]   {line}")
                                    
                                    # If dghs-imgutils fails, try alternatives
                                    if "dghs-imgutils" in pkg:
                                        print("[Danbooru Lookup] Trying alternative install methods...")
                                        # Try without version
                                        alt_cmd = [sys.executable, "-m", "pip", "install", "dghs-imgutils"]
                                        alt_result = subprocess.run(alt_cmd, capture_output=True, text=True)
                                        if alt_result.returncode == 0:
                                            print("[Danbooru Lookup] Installed dghs-imgutils (no version)")
                                        else:
                                            # Try with --no-deps to avoid cv2 issues
                                            alt_cmd = [sys.executable, "-m", "pip", "install", "--no-deps", "dghs-imgutils"]
                                            alt_result = subprocess.run(alt_cmd, capture_output=True, text=True)
                                            if alt_result.returncode == 0:
                                                print("[Danbooru Lookup] Installed dghs-imgutils (no deps)")
                                                # Now install essential dependencies that won't conflict
                                                print("[Danbooru Lookup] Installing imgutils dependencies...")
                                                essential_deps = ["huggingface-hub", "onnxruntime", "pillow", "hbutils"]
                                                for edep in essential_deps:
                                                    try:
                                                        cmd = [sys.executable, "-m", "pip", "install", edep]
                                                        result = subprocess.run(cmd, capture_output=True, text=True)
                                                        if result.returncode == 0:
                                                            print(f"[Danbooru Lookup]   Installed {edep}")
                                                    except:
                                                        pass
                            except Exception as e:
                                print(f"[Danbooru Lookup] Error installing {pkg}: {e}")
                        
                        # Try JAX last
                        if "jax/flax" in missing_deps:
                            for pkg in ["jax", "jaxlib", "flax"]:
                                try:
                                    cmd = [sys.executable, "-m", "pip", "install", pkg]
                                    result = subprocess.run(cmd, capture_output=True, text=True)
                                    if result.returncode == 0:
                                        print(f"[Danbooru Lookup] Installed {pkg}")
                                except:
                                    pass
                        
                        # Final check - see what's actually available now
                        print("[Danbooru Lookup] Checking what was actually installed...")
                        actually_installed = []
                        already_had = []
                        
                        try:
                            import cv2
                            already_had.append("opencv")
                        except: pass
                        
                        try:
                            import faiss
                            actually_installed.append("faiss")
                        except: pass
                        try:
                            import imgutils
                            actually_installed.append("dghs-imgutils")
                        except: 
                            try:
                                from imgutils.tagging import wd14
                                actually_installed.append("dghs-imgutils")
                            except: pass
                        try:
                            import jax
                            actually_installed.append("jax")
                        except: pass
                        
                        if already_had:
                            print(f"[Danbooru Lookup] Already available: {', '.join(already_had)}")
                        
                        if actually_installed:
                            print(f"[Danbooru Lookup] Successfully available: {', '.join(actually_installed)}")
                            install_success = True  # Partial success
                        else:
                            install_success = False
                    else:
                        if result.stderr:
                            print(f"[Danbooru Lookup] Error: {result.stderr}")
                        install_success = False
                        
            except Exception as e:
                print(f"[Danbooru Lookup] Installation error: {e}")
                install_success = False
            
            if install_success:
                # Try to reload modules to pick up newly installed packages
                try:
                    import importlib
                    # Reload modules to pick up newly installed packages
                    modules_to_reload = [
                        'comfyui-danbooru-lookup.modules.tag_embeddings',
                        'comfyui-danbooru-lookup.modules.wd14_embeddings',
                        'comfyui-danbooru-lookup.modules.danbooru_lookup_advanced'
                    ]
                    for module in modules_to_reload:
                        if module in sys.modules:
                            importlib.reload(sys.modules[module])
                    print("[Danbooru Lookup] Reloaded modules to detect newly installed packages.")
                    # Re-check ALL dependencies
                    import faiss
                    import pandas
                    import numpy
                    import tqdm
                    from imgutils.tagging import wd14
                    import PIL
                    import jax
                    import flax
                    DEPENDENCIES_INSTALLED = True
                    print("[Danbooru Lookup] Dependencies installed and verified successfully!")
                except:
                    print("[Danbooru Lookup] Dependencies installed but require restart to fully activate.")
                        
        except Exception as install_error:
            print(f"[Danbooru Lookup] Installation error: {install_error}")
    
    if not DEPENDENCIES_INSTALLED:
        print("[Danbooru Lookup] Dependencies will be installed automatically on next restart.")
        print("[Danbooru Lookup] Please restart ComfyUI.")
        print("[Danbooru Lookup] If you're seeing old error messages, Python cache may need clearing.")
        print("[Danbooru Lookup] You can also run: python manual_install.py")

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
                "Please restart ComfyUI to install dependencies automatically."
            )
            print(f"[Danbooru Lookup] {error_msg}")
            return ("ERROR: Dependencies not installed - Please restart ComfyUI",)
    
    NODE_CLASS_MAPPINGS = {
        "DanbooruFAISSLookup": DanbooruFAISSLookupStub
    }
    
    NODE_DISPLAY_NAME_MAPPINGS = {
        "DanbooruFAISSLookup": "Danbooru FAISS Lookup (INSTALL REQUIRED)"
    }
else:
    # Dependencies are installed, load the real nodes
    try:
        from .modules.danbooru_lookup import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        
        # Try to load advanced node if additional dependencies are available
        try:
            from .modules.danbooru_lookup_advanced import NODE_CLASS_MAPPINGS as ADV_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as ADV_DISPLAY
            NODE_CLASS_MAPPINGS.update(ADV_MAPPINGS)
            NODE_DISPLAY_NAME_MAPPINGS.update(ADV_DISPLAY)
            print("[Danbooru Lookup] Advanced node loaded successfully")
        except ImportError as e:
            print(f"[Danbooru Lookup] Advanced node not available: {e}")
            print("[Danbooru Lookup] Basic node is still available")
        
        # Try to load WD14 to Conditioning node
        try:
            from .modules.wd14_to_conditioning import NODE_CLASS_MAPPINGS as WD14_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as WD14_DISPLAY
            NODE_CLASS_MAPPINGS.update(WD14_MAPPINGS)
            NODE_DISPLAY_NAME_MAPPINGS.update(WD14_DISPLAY)
            print("[Danbooru Lookup] WD14 to Conditioning node loaded successfully")
        except ImportError as e:
            print(f"[Danbooru Lookup] WD14 to Conditioning node not available: {e}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load Danbooru FAISS Lookup node: {e}")
        traceback.print_exc()
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]