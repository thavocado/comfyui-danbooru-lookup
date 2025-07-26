#!/bin/bash

echo "============================================================"
echo "ComfyUI Danbooru FAISS Lookup - Dependency Installer"
echo "============================================================"
echo ""

# Try to find the appropriate Python command
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "Error: Python not found. Please install Python 3."
    exit 1
fi

echo "Using Python: $PYTHON_CMD"

# Run the Python install script
$PYTHON_CMD install.py

echo ""
echo "Installation complete. Press Enter to exit..."
read -r