import os
import requests
import logging
import json
from pathlib import Path

# Try to import tqdm, but don't fail if it's not available
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False

class DataLoader:
    """Handle downloading of required data files from HuggingFace."""
    
    def __init__(self, data_dir):
        self.data_dir = Path(data_dir)
        self.base_url = "https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground/resolve/main"
        
        # Files to download
        self.required_files = {
            "index/cosine_ids.npy": {
                "url": f"{self.base_url}/index/cosine_ids.npy"
            },
            "index/cosine_knn.index": {
                "url": f"{self.base_url}/index/cosine_knn.index"
            },
            "index/cosine_infos.json": {
                "url": f"{self.base_url}/index/cosine_infos.json"
            },
            "data/selected_tags.csv": {
                "url": f"{self.base_url}/data/selected_tags.csv"
            },
        }
    
    def ensure_data_downloaded(self, force_redownload=False):
        """Check if all required files exist and are valid, download if missing or corrupted."""
        files_to_download = []
        
        for relative_path, file_info in self.required_files.items():
            local_path = self.data_dir / relative_path
            
            # Check if file needs to be downloaded
            if force_redownload or not local_path.exists():
                files_to_download.append((relative_path, file_info))
            elif local_path.exists():
                # Validate existing file
                if not self._validate_file(local_path):
                    logging.warning(f"File {relative_path} appears to be corrupted. Re-downloading...")
                    files_to_download.append((relative_path, file_info))
        
        if files_to_download:
            logging.info(f"Need to download {len(files_to_download)} files from HuggingFace...")
            for relative_path, file_info in files_to_download:
                local_path = self.data_dir / relative_path
                url = file_info['url']
                
                # Try downloading with retries
                success = False
                for attempt in range(3):  # 3 attempts
                    try:
                        if attempt > 0:
                            logging.info(f"Retry attempt {attempt + 1} for {relative_path}...")
                        
                        self._download_file(url, local_path)
                        
                        # Validate downloaded file
                        if self._validate_file(local_path):
                            success = True
                            break
                        else:
                            logging.warning(f"Downloaded file {relative_path} failed validation.")
                            if local_path.exists():
                                local_path.unlink()  # Delete corrupted file
                    except Exception as e:
                        logging.error(f"Download attempt {attempt + 1} failed: {e}")
                        if local_path.exists():
                            local_path.unlink()  # Delete partial file
                
                if not success:
                    raise RuntimeError(f"Failed to download {relative_path} after 3 attempts")
            
            logging.info("All required files downloaded and validated successfully.")
    
    def _validate_file(self, file_path):
        """Validate that a file exists and is not empty."""
        if not file_path.exists():
            return False
        
        actual_size = file_path.stat().st_size
        
        # Check it's not empty
        if actual_size == 0:
            logging.warning(f"File {file_path.name} is empty")
            return False
        
        return True
    
    def _download_file(self, url, destination):
        """Download a file from URL to destination with progress bar."""
        destination.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(destination, 'wb') as f:
                if total_size == 0:
                    # No content length header
                    f.write(response.content)
                else:
                    # Show progress bar if tqdm is available
                    chunk_size = 8192
                    if HAS_TQDM:
                        with tqdm(total=total_size, unit='B', unit_scale=True, desc=destination.name) as pbar:
                            for chunk in response.iter_content(chunk_size=chunk_size):
                                if chunk:
                                    f.write(chunk)
                                    pbar.update(len(chunk))
                    else:
                        # No progress bar, just download
                        downloaded = 0
                        for chunk in response.iter_content(chunk_size=chunk_size):
                            if chunk:
                                f.write(chunk)
                                downloaded += len(chunk)
                                if downloaded % (1024 * 1024) == 0:  # Print every MB
                                    print(f"  Downloaded {downloaded // (1024 * 1024)}MB of {total_size // (1024 * 1024)}MB...")
            
            # Verify file was downloaded completely
            if destination.exists():
                actual_size = destination.stat().st_size
                if total_size > 0 and actual_size != total_size:
                    raise RuntimeError(f"Incomplete download: {actual_size} != {total_size} bytes")
            
            logging.info(f"Downloaded {destination.name} ({actual_size} bytes)")
            
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            if destination.exists():
                destination.unlink()  # Remove partial download
            raise