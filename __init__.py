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
    import msgpack
except ImportError:
    missing_deps.append("msgpack")

try:
    import jax
    import flax
except ImportError as e:
    missing_deps.append("jax/flax")
    print(f"[Danbooru Lookup] Initial JAX/FLAX detection failed: {e}")

if missing_deps:
    DEPENDENCIES_INSTALLED = False
    print(f"[Danbooru Lookup] Missing dependencies detected: {', '.join(missing_deps)}")
    print("[Danbooru Lookup] Attempting to install dependencies...")
    
    # Try to install dependencies directly
    requirements_file = Path(__file__).parent / "requirements.txt"
    if requirements_file.exists():
        try:
            print(f"[Danbooru Lookup] Installing dependencies...")
            
            # Use --no-cache-dir to avoid stale cached packages
            cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "-r", str(requirements_file)]
            
            install_success = False
            try:
                print(f"[Danbooru Lookup] Running: {' '.join(cmd)}")
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # Show pip output for debugging
                if result.stdout:
                    print("[Danbooru Lookup] pip output:")
                    for line in result.stdout.strip().split('\n')[:10]:  # Show first 10 lines
                        print(f"[Danbooru Lookup]   {line}")
                    if len(result.stdout.strip().split('\n')) > 10:
                        print("[Danbooru Lookup]   ... (output truncated)")
                
                if result.returncode == 0:
                    print("[Danbooru Lookup] pip install completed successfully!")
                    
                    # Verify what was actually installed
                    try:
                        verify_cmd = [sys.executable, "-m", "pip", "list"]
                        verify_result = subprocess.run(verify_cmd, capture_output=True, text=True)
                        if verify_result.returncode == 0:
                            installed_packages = verify_result.stdout.lower()
                            critical_packages = ["msgpack", "jax ", "jaxlib", "flax", "faiss"]
                            missing_critical = []
                            for pkg in critical_packages:
                                if pkg not in installed_packages:
                                    missing_critical.append(pkg.strip())
                            
                            if missing_critical:
                                print(f"[Danbooru Lookup] WARNING: Critical packages not found in pip list: {', '.join(missing_critical)}")
                                print("[Danbooru Lookup] Will attempt individual installation...")
                                install_success = False  # Force fallback installation
                            else:
                                install_success = True
                    except:
                        install_success = True  # Assume success if we can't verify
                else:
                    print(f"[Danbooru Lookup] Installation failed with code {result.returncode}")
                    if result.stderr:
                        print(f"[Danbooru Lookup] Installation stderr: {result.stderr[:500]}")
                    
                    if result.stderr and "Access is denied" in result.stderr:
                        print("[Danbooru Lookup] Access denied error - some packages are in use")
                        print("[Danbooru Lookup] Trying to install missing packages only...")
                        
                        # Install msgpack first if it's missing (JAX dependency)
                        if "msgpack" in missing_deps:
                            print("[Danbooru Lookup] Installing msgpack first (required by JAX)...")
                            try:
                                cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "msgpack"]
                                result = subprocess.run(cmd, capture_output=True, text=True)
                                if result.returncode == 0:
                                    print("[Danbooru Lookup]   Installed msgpack")
                                else:
                                    print(f"[Danbooru Lookup]   Failed to install msgpack: {result.stderr}")
                            except Exception as e:
                                print(f"[Danbooru Lookup]   Error installing msgpack: {e}")
                        
                        # Try installing only the missing packages
                        for dep in missing_deps:
                            if dep == "msgpack":
                                continue  # Already handled above
                            elif dep == "dghs-imgutils":
                                pkg = "dghs-imgutils>=0.17.0"
                            elif dep == "jax/flax":
                                # JAX needs special handling - install components separately
                                print("[Danbooru Lookup] Installing JAX/FLAX components...")
                                jax_packages = ["jax", "jaxlib", "flax"]
                                jax_success = True
                                for jax_pkg in jax_packages:
                                    try:
                                        cmd = [sys.executable, "-m", "pip", "install", jax_pkg]
                                        result = subprocess.run(cmd, capture_output=True, text=True)
                                        if result.returncode == 0:
                                            print(f"[Danbooru Lookup]   Installed {jax_pkg}")
                                        else:
                                            print(f"[Danbooru Lookup]   Failed to install {jax_pkg}")
                                            jax_success = False
                                    except Exception as e:
                                        print(f"[Danbooru Lookup]   Error installing {jax_pkg}: {e}")
                                        jax_success = False
                                if jax_success:
                                    install_success = True
                                continue  # Skip to next dependency
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
                        
                        # Try JAX last if not already handled
                        if "jax/flax" in missing_deps and not install_success:
                            print("[Danbooru Lookup] Attempting JAX/FLAX installation as last resort...")
                            jax_installed = 0
                            for pkg in ["jax", "jaxlib", "flax"]:
                                try:
                                    cmd = [sys.executable, "-m", "pip", "install", pkg]
                                    result = subprocess.run(cmd, capture_output=True, text=True)
                                    if result.returncode == 0:
                                        print(f"[Danbooru Lookup] Installed {pkg}")
                                        jax_installed += 1
                                    else:
                                        print(f"[Danbooru Lookup] Failed to install {pkg}")
                                        if result.stderr:
                                            error_lines = result.stderr.strip().split('\n')[:2]
                                            for line in error_lines:
                                                print(f"[Danbooru Lookup]   {line}")
                                except Exception as e:
                                    print(f"[Danbooru Lookup] Error installing {pkg}: {e}")
                            
                            if jax_installed == 3:
                                install_success = True
                        
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
                            print(f"[Danbooru Lookup] Error: {result.stderr[:500]}")
                        install_success = False
                
                # If main install failed OR critical packages are missing, try individual installs
                if not install_success and missing_deps:
                    print("[Danbooru Lookup] Attempting individual package installation...")
                    
                    # Always install msgpack first if needed
                    if "msgpack" in missing_deps:
                        print("[Danbooru Lookup] Installing msgpack (required by JAX)...")
                        try:
                            cmd = [sys.executable, "-m", "pip", "install", "--no-cache-dir", "msgpack"]
                            result = subprocess.run(cmd, capture_output=True, text=True)
                            if result.returncode == 0:
                                print("[Danbooru Lookup]   ✓ msgpack installed")
                            else:
                                print(f"[Danbooru Lookup]   ✗ msgpack failed: {result.stderr[:200]}")
                        except Exception as e:
                            print(f"[Danbooru Lookup]   Error: {e}")
                        
            except Exception as e:
                print(f"[Danbooru Lookup] Installation error: {e}")
                install_success = False
            
            if install_success:
                # Try to reload modules to pick up newly installed packages
                print("[Danbooru Lookup] Verifying installed dependencies...")
                verification_failed = []
                
                try:
                    # Clear import cache for these modules
                    import importlib
                    import importlib.util
                    
                    # Force reimport of key modules - including submodules
                    modules_to_remove = []
                    for mod_name in list(sys.modules.keys()):
                        if mod_name.startswith(('jax.', 'jax_', 'flax.', 'faiss.', 'imgutils.')):
                            modules_to_remove.append(mod_name)
                        elif mod_name in ['jax', 'flax', 'faiss', 'imgutils']:
                            modules_to_remove.append(mod_name)
                    
                    for mod in modules_to_remove:
                        if mod in sys.modules:
                            del sys.modules[mod]
                    
                    if modules_to_remove:
                        print(f"[Danbooru Lookup] Cleared {len(modules_to_remove)} cached modules")
                    
                    # Verify each dependency individually
                    try:
                        import faiss
                        print("[Danbooru Lookup] ✓ faiss verified")
                    except ImportError as e:
                        verification_failed.append("faiss")
                        print(f"[Danbooru Lookup] faiss import error: {e}")
                    
                    try:
                        import pandas
                        print("[Danbooru Lookup] ✓ pandas verified")
                    except ImportError:
                        verification_failed.append("pandas")
                    
                    try:
                        import numpy
                        print("[Danbooru Lookup] ✓ numpy verified")
                    except ImportError:
                        verification_failed.append("numpy")
                    
                    try:
                        import tqdm
                        print("[Danbooru Lookup] ✓ tqdm verified")
                    except ImportError:
                        verification_failed.append("tqdm")
                    
                    try:
                        from imgutils.tagging import wd14
                        print("[Danbooru Lookup] ✓ dghs-imgutils verified")
                    except ImportError as e:
                        verification_failed.append("dghs-imgutils")
                        print(f"[Danbooru Lookup] dghs-imgutils import error: {e}")
                    
                    try:
                        import PIL
                        print("[Danbooru Lookup] ✓ PIL verified")
                    except ImportError:
                        verification_failed.append("PIL")
                    
                    try:
                        # Check if JAX is already loaded to avoid PyTreeDef re-registration
                        jax_already_loaded = 'jax' in sys.modules and hasattr(sys.modules['jax'], '__version__')
                        
                        import jax
                        import jax.numpy as jnp
                        import flax
                        
                        # Only print success if this is a fresh import
                        if not jax_already_loaded:
                            print("[Danbooru Lookup] ✓ JAX/FLAX verified")
                        else:
                            print("[Danbooru Lookup] ✓ JAX/FLAX already loaded")
                            
                    except ImportError as e:
                        verification_failed.append("jax/flax")
                        print(f"[Danbooru Lookup] JAX/FLAX import error: {e}")
                        import traceback
                        traceback.print_exc()
                    except Exception as e:
                        # Catch non-ImportError exceptions like PyTreeDef registration
                        print(f"[Danbooru Lookup] JAX/FLAX initialization error ({type(e).__name__}): {e}")
                        # Don't add to verification_failed if it's just a re-registration issue
                        if "PyTreeDef" not in str(e):
                            verification_failed.append("jax/flax")
                    
                    if not verification_failed:
                        DEPENDENCIES_INSTALLED = True
                        print("[Danbooru Lookup] All dependencies installed and verified successfully!")
                    else:
                        print(f"[Danbooru Lookup] Failed to verify: {', '.join(verification_failed)}")
                        print("[Danbooru Lookup] Dependencies installed but require restart to fully activate.")
                        
                except Exception as e:
                    print(f"[Danbooru Lookup] Verification error ({type(e).__name__}): {e}")
                    if "PyTreeDef" in str(e):
                        print("[Danbooru Lookup] JAX is already initialized. This is usually fine.")
                        # If only JAX had the PyTreeDef issue but everything else verified, consider it success
                        if not verification_failed or verification_failed == ["jax/flax"]:
                            DEPENDENCIES_INSTALLED = True
                            print("[Danbooru Lookup] Dependencies verified (JAX already loaded).")
                        else:
                            print("[Danbooru Lookup] Dependencies installed but require restart to fully activate.")
                    else:
                        print("[Danbooru Lookup] Dependencies installed but require restart to fully activate.")
                        import traceback
                        traceback.print_exc()
                        
        except Exception as install_error:
            print(f"[Danbooru Lookup] Installation error: {install_error}")
    
    if not DEPENDENCIES_INSTALLED:
        print("[Danbooru Lookup] Dependencies will be installed automatically on next restart.")
        print("[Danbooru Lookup] Please restart ComfyUI.")
        
        # For portable installations, provide more specific help
        if "python_embeded" in sys.executable or "python_embedded" in sys.executable:
            print("[Danbooru Lookup] Detected portable ComfyUI installation.")
            print("[Danbooru Lookup] If installation keeps failing:")
            print("[Danbooru Lookup]   1. Close ComfyUI completely")
            print("[Danbooru Lookup]   2. Delete __pycache__ folders in custom_nodes/comfyui-danbooru-lookup/")
            print("[Danbooru Lookup]   3. Run: python_embeded\\python.exe -m pip install jax jaxlib flax")
            print("[Danbooru Lookup]   4. Restart ComfyUI")
        
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
        except ValueError as e:
            if "PyTreeDef" in str(e):
                print("[Danbooru Lookup] JAX PyTreeDef conflict detected. This happens when JAX is already loaded.")
                print("[Danbooru Lookup] The advanced node may have limited functionality.")
                print("[Danbooru Lookup] Basic node is still available")
            else:
                print(f"[Danbooru Lookup] Error loading advanced node: {e}")
                print("[Danbooru Lookup] Basic node is still available")
        
        # Try to load WD14 to Conditioning node
        try:
            from .modules.wd14_to_conditioning import NODE_CLASS_MAPPINGS as WD14_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS as WD14_DISPLAY
            NODE_CLASS_MAPPINGS.update(WD14_MAPPINGS)
            NODE_DISPLAY_NAME_MAPPINGS.update(WD14_DISPLAY)
            print("[Danbooru Lookup] WD14 to Conditioning node loaded successfully")
        except ImportError as e:
            print(f"[Danbooru Lookup] WD14 to Conditioning node not available: {e}")
        except ValueError as e:
            if "PyTreeDef" in str(e):
                print("[Danbooru Lookup] JAX conflict affects WD14 node. Basic functionality still available.")
            else:
                print(f"[Danbooru Lookup] Error loading WD14 node: {e}")
            
    except Exception as e:
        print(f"[ERROR] Failed to load Danbooru FAISS Lookup node: {e}")
        traceback.print_exc()
        NODE_CLASS_MAPPINGS = {}
        NODE_DISPLAY_NAME_MAPPINGS = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]