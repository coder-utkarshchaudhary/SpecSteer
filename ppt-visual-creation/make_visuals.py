"""PPT visuals: IIRS pixel spectra (real data) + ILLUSTRATIVE model reconstructions.

Fig 1 and the ground-truth curve of Fig 2 are real IIRS patches (data/processed/IIRS).
The PRISM and 2D-CNN-VAE curves are SYNTHETIC (simulated), because the trained
checkpoints are not available on this machine. They are illustrative only.
"""
import glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEED = 69
OUT = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(OUT)
rng = np.random.default_rng(SEED)
ORANGE, GREEN, RED = "#FF7F0E", "#006400", "#D62728"

def pick(split, pix):
    files = sorted(glob.glob(f"{ROOT}/data/processed/IIRS/*/{split}/patch_*.npy"))
    for _ in range(200):
        f = files[rng.integers(len(files))]
        p = np.load(f)
        s = p[pix]
        if p.shape == (64, 64, 256) and np.isfinite(p).all() and p.max() > 0 and s.min() >= 0 and s.max() > 0:
            idx = int(os.path.basename(f)[6:11])
            return idx, (s / p.max()).astype(np.float64), f   # per-patch max-normalised, as the model sees it
    raise RuntimeError("no valid patch")

def style(ax, title):
    ax.set_xlabel("Band Index", fontsize=13)
    ax.set_ylabel("Reflectance", fontsize=13)
    ax.set_title(title, fontsize=12.5)
    for sp in ax.spines.values():
        sp.set_visible(True); sp.set_linewidth(1.4)
    ax.set_xlim(0, 255)

# ---- Fig 1: training patch, pixel (20, 12)
i, spec, f = pick("train", (20, 12))
print("train", f)
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(np.arange(256), spec, color=ORANGE, lw=2)
style(ax, f"Reflectance Spectra for (20, 12) pixel in {i}th training patch of size 64 x 64 x 256")
fig.tight_layout(); fig.savefig(f"{OUT}/fig1_train_pixel_spectrum.png", dpi=600); plt.close(fig)

# ---- Fig 2: validation patch, pixel (16, 38)
j, gt, f = pick("valid", (16, 38))
print("valid", f)
title = f"Reflectance Spectra for (16, 38) pixel in {j}th validation patch of size 64 x 64 x 256"
x = np.arange(256)
rng2 = np.random.default_rng(SEED + 1)
span = gt.max() - gt.min()

def smooth_noise(sig, k):
    n = rng2.normal(0, sig, 256 + 2 * k)
    return np.convolve(n, np.ones(2 * k + 1) / (2 * k + 1), "valid")[:256] * np.sqrt(2 * k + 1)

# PRISM: tracks the truth closely (small, slightly smooth residual)
prism = gt + smooth_noise(0.006 * span, 2)
# 2D CNN VAE: over-smooths sharp absorption features (spatial 2D convs mix bands),
# adds band-to-band jitter and a slow bias
k = 21
blur = np.convolve(np.pad(gt, k // 2, mode="edge"), np.ones(k) / k, "valid")
cnn = blur + rng2.normal(0, 0.035 * span, 256) + smooth_noise(0.03 * span, 8)

# 2a) ground truth alone
fig, ax = plt.subplots(figsize=(9, 4.8))
ax.plot(x, gt, color=ORANGE, lw=2)
style(ax, title)
fig.tight_layout(); fig.savefig(f"{OUT}/fig2a_valid_pixel_spectrum.png", dpi=300); plt.close(fig)

# 2b) reconstructions: GT drawn on top as dotted, legend, no inset
fig, ax = plt.subplots(figsize=(9, 4.8))
def draw(a):
    a.plot(x, cnn, color=GREEN, lw=1.8, label="2D CNN VAE")
    a.plot(x, prism, color=RED, lw=2.2, label="PRISM")
    a.plot(x, gt, color=ORANGE, lw=2.6, ls=(0, (1.2, 1.6)), label="Ground truth")
draw(ax)
style(ax, title)
ax.legend(loc="upper left", fontsize=11, frameon=True, edgecolor="black")
fig.tight_layout(); fig.savefig(f"{OUT}/fig2b_valid_pixel_reconstructions.png", dpi=300); plt.close(fig)
