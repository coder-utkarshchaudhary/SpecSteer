"""
inference/downstream_variants.py
--------------------------------
Custom downstream-evaluation entry-point that registers the new `vae-our` variants
(NL, SpecViT, and NL-SpecViT hybrid) and delegates execution to the
standard downstream main loop in downstream.py.
"""
import sys
import modules.vae_our_variants  # Dynamically registers the custom model variants in the MODELS registry
from inference.downstream import main

if __name__ == "__main__":
    main()
