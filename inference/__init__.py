"""Evaluation package: reconstruction metrics, latent probes, and statistics.

`__init__.py` is required here, not optional: `inference/inference.py` has the
same name as its own package, so without it `import inference.inference` from a
script inside this directory resolves `inference` to the MODULE and fails with
"'inference' is not a package". The modules here also insert the repo root at
sys.path[0] so the package wins over the sibling module.
"""
