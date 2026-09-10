"""
train/train_variants.py
-----------------------
Custom training entry-point that registers the new `vae-our` variants
(NL, SpecViT, and NL-SpecViT hybrid) and delegates execution to the
standard main training loop in train.py.
"""
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) in sys.path:
    sys.path.remove(str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT))

import modules.vae_our_variants  # Dynamically registers the custom model variants in the MODELS registry
from train.train import main

if __name__ == "__main__":
    main()
