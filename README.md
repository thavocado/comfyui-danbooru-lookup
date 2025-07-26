# ComfyUI Danbooru FAISS Lookup

A ComfyUI custom node that performs FAISS cosine similarity lookup on Danbooru embeddings using CLIP conditioning inputs.

## Features

- Takes positive and negative CLIP conditioning as inputs
- Performs FAISS similarity search on Danbooru image embeddings
- Returns the best matching Danbooru post ID
- Supports rating filtering (General, Sensitive, Questionable, Explicit)
- Optional Danbooru API authentication for better access
- Automatic download of required data files from HuggingFace

## Installation

1. Clone this repository into your ComfyUI custom_nodes folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/thavocado/comfyui-danbooru-lookup
   ```

2. Install the required dependencies (if auto install fails):
   ```bash
   pip install -r comfyui-danbooru-lookup/requirements.txt
   ```

3. Restart ComfyUI

The node will automatically download required data files (~1GB) from HuggingFace on first use.

## Usage

1. Find the node under the "conditioning/danbooru" category
2. Connect positive and negative CLIP Text Encode nodes to the conditioning inputs
3. Configure optional parameters:
   - `selected_ratings`: Comma-separated list of ratings to include (default: "General,Sensitive")
   - `n_neighbours`: Number of similar images to search (default: 5)
   - `api_username`: Optional Danbooru API username
   - `api_key`: Optional Danbooru API key

The node outputs a Danbooru post ID as a string that can be used with other nodes or for reference.

## Data Source

This node uses embeddings and indices from the [Danbooru2022 Embeddings Playground](https://huggingface.co/spaces/SmilingWolf/danbooru2022_embeddings_playground) by SmilingWolf.

## Requirements

- ComfyUI
- Python 3.8+
- ~1GB disk space for data files

## License

This project is released under the MIT License. 