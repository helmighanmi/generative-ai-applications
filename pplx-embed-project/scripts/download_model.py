#!/usr/bin/env python3
"""
Pre-downloads pplx-embed-v1-0.6B weights into ./model/
Run this ONCE before starting the API.
"""

import os
import sys
import shutil
from pathlib import Path

MODEL_NAME = "perplexity-ai/pplx-embed-v1-0.6B"
PROJECT_ROOT = Path(__file__).parent.parent
SAVE_DIR = PROJECT_ROOT / "model"

print(f"⬇️  Downloading '{MODEL_NAME}' into '{SAVE_DIR}' ...")
print("   This requires ~1.2 GB of disk space and an internet connection.\n")

try:
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import snapshot_download
except ImportError:
    print("❌ Required packages not installed. Run: pip install sentence-transformers huggingface_hub")
    sys.exit(1)

# Create model directory
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Download the full model from HuggingFace Hub including all custom files
print("   Downloading from HuggingFace Hub (including custom modules)...")
snapshot_download(
    repo_id=MODEL_NAME,
    local_dir=str(SAVE_DIR),
    local_dir_use_symlinks=False,
    resume_download=True,
)

print(f"\n✅ Model downloaded successfully to: {SAVE_DIR}")

# Verify st_quantize.py exists
st_quantize_path = SAVE_DIR / "st_quantize.py"
if st_quantize_path.exists():
    print("   ✓ Custom module 'st_quantize.py' found")
else:
    print("   ⚠️  Warning: 'st_quantize.py' not found. Model may not load correctly.")

print("\nYou can now run the API offline:")
print("   python api/main.py")