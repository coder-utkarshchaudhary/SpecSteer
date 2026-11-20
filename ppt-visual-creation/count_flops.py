"""Forward-pass FLOPs per sample (batch 1, inference) for each model x dataset.
FLOPs = 2 x MACs, counted by torch FlopCounterMode (convs + matmuls).
Run from repo root:  PYTHONPATH=. .venv/bin/python ppt-visual-creation/count_flops.py
"""
import sys, json, importlib.util, torch
from torch.utils.flop_counter import FlopCounterMode
from utils.config import settings, apply_dataset
from utils.hyperparams import apply_hyperparams, load_hyperparams
from modules.registry import build_model

torch.set_num_threads(2)
MODELS = ["vae-our", "vae-our-nl", "vae-standard", "vae-3d-spatio-spectral", "vae-1d-pixelwise"]
DS = ["IIRS", "M3", "AVIRIS", "CRIMS"]
res = {}
for ds in DS:
    for m in MODELS:
        apply_dataset(ds); apply_hyperparams(settings, load_hyperparams(ds))
        model = build_model(m).eval()
        x = torch.randn(1, settings.input_height, settings.input_width, settings.input_channels)
        with torch.no_grad():
            model(x)                       # materialise LazyLinear
            fc = FlopCounterMode(display=False)
            with fc:
                model(x)
        res[f"{m}|{ds}"] = fc.get_total_flops()
        print(f"{ds:7s} {m:24s} {res[f'{m}|{ds}']/1e9:10.2f} GFLOPs", flush=True)
        del model
json.dump(res, open("ppt-visual-creation/flops.json", "w"), indent=1)
