"""
inference/inference_variants.py
-------------------------------
Custom inference/evaluation entry-point that registers the new `vae-our` variants
(NL, SpecViT, and NL-SpecViT hybrid) and delegates execution to the
standard main inference loop in inference.py.
"""
import sys
import modules.vae_our_variants  # Dynamically registers the custom model variants in the MODELS registry
from inference.inference import main

if __name__ == "__main__":
    main()
