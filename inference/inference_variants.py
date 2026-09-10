"""
inference/inference_variants.py
-------------------------------
Custom inference/evaluation entry-point that registers the new `vae-our` variants
(NL, SpecViT, and NL-SpecViT hybrid) and delegates execution to the
standard main inference loop in inference.py.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# Clean sys.path to avoid name shadowing (e.g. inference_variants.py parent 'inference' folder shadowing the 'inference' package)
while str(SCRIPT_DIR) in sys.path:
    sys.path.remove(str(SCRIPT_DIR))
while str(SCRIPT_DIR.resolve()) in sys.path:
    sys.path.remove(str(SCRIPT_DIR.resolve()))

# Insert repo root as index 0 to ensure the repository package is imported correctly
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
else:
    sys.path.remove(str(REPO_ROOT))
    sys.path.insert(0, str(REPO_ROOT))

import modules.vae_our_variants  # Dynamically registers the custom model variants in the MODELS registry
from inference.inference import main

if __name__ == "__main__":
    main()
