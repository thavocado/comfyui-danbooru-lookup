import os
import requests
import logging
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
        
        self.required_files = {
            "index/cosine_ids.npy": f"{self.base_url}/index/cosine_ids.npy",
            "index/cosine_knn.index": f"{self.base_url}/index/cosine_knn.index",
            "index/cosine_infos.json": f"{self.base_url}/index/cosine_infos.json",
            "data/selected_tags.csv": f"{self.base_url}/data/selected_tags.csv",
        }
    
    def ensure_data_downloaded(self):
        """Check if all required files exist, download if missing."""
        missing_files = []
        
        for relative_path in self.required_files:
            local_path = self.data_dir / relative_path
            if not local_path.exists():
                missing_files.append(relative_path)
        
        if missing_files:
            logging.info(f"Missing {len(missing_files)} required files. Downloading from HuggingFace...")
            for relative_path in missing_files:
                url = self.required_files[relative_path]
                local_path = self.data_dir / relative_path
                self._download_file(url, local_path)
            logging.info("All required files downloaded successfully.")
    
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
            
            logging.info(f"Downloaded {destination.name}")
            
        except Exception as e:
            logging.error(f"Failed to download {url}: {e}")
            raise