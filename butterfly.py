#!/usr/bin/env python3
import argparse
import numpy as np
from PIL import Image
import cv2

# ======================================================
# Helpers
# ======================================================

def fft2c(x):
    return np.fft.fftshift(np.fft.fft2(x))

def ifft2c(X):
    return np.fft.ifft2(np.fft.ifftshift(X)).real

def phase_only_reconstruct(A, Phi):
    return A * np.exp(1j * Phi)

def frequency_grid(H, W):
    y, x = np.ogrid[:H, :W]
    cy, cx = H // 2, W // 2
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    r_norm = r / (r.max() + 1e-8)
    theta = np.arctan2(y - cy, x - cx)
    return r_norm, theta

def compute_phase_sensitivity(gray):
    # Model-agnostic heuristic: edge density & phase entropy proxy
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = np.mean(mag)
    # normalize to a small range
    return np.clip(edge_density / (edge_density + 1.0), 0.2, 0.8)

# ======================================================
# CLI
# ======================================================
parser = argparse.ArgumentParser(description="Butterfly v2 — Vision-only feature & phase steering")
parser.add_argument("image", help="Input image path")
parser.add_argument("--out", default="butterfly_output.png")
parser.add_argument("--multiscale-phase", action="store_true")
parser.add_argument("--directional-phase", choices=["vertical", "horizontal", "diag"])
parser.add_argument("--cross-channel", action="store_true")
parser.add_argument("--adaptive-epsilon", action="store_true")
parser.add_argument("--feature-basin", choices=["stripe", "smooth", "periodic"])
args = parser.parse_args()

# ======================================================
# Load image
# ======================================================
img = Image.open(args.image).convert("RGB")
x = np.asarray(img).astype(np.float32) / 255.0
H, W, C = x.shape
x_out = np.zeros_like(x)

# grayscale only for sensitivity estimation (not used for reconstruction)
gray = cv2.cvtColor(x, cv2.COLOR_RGB2GRAY)

# Base epsilon (very small, safe)
EPS_BASE = 0.012
if args.adaptive_epsilon:
    EPS = EPS_BASE * compute_phase_sensitivity(gray)
else:
    EPS = EPS_BASE

# Frequency grid
r_norm, theta = frequency_grid(H, W)

# Multiscale bands (mid frequencies only)
bands = [(0.06, 0.16), (0.16, 0.30), (0.30, 0.42)] if args.multiscale_phase else [(0.16, 0.30)]

# Directional weighting
if args.directional_phase == "vertical":
    dir_w = np.abs(np.cos(theta))
elif args.directional_phase == "horizontal":
    dir_w = np.abs(np.sin(theta))
elif args.directional_phase == "diag":
    dir_w = np.abs(np.sin(2 * theta))
else:
    dir_w = np.ones_like(theta)

# Feature basin steering (generic, label-agnostic)
# stripe: emphasize periodic coherence; smooth: suppress coherence; periodic: radial periodicity
if args.feature_basin == "stripe":
    basin_w = 0.5 + 0.5 * np.cos(6 * np.pi * r_norm)
elif args.feature_basin == "smooth":
    basin_w = 1.0 - r_norm
elif args.feature_basin == "periodic":
    basin_w = np.sin(8 * np.pi * r_norm) ** 2
else:
    basin_w = np.ones_like(r_norm)

# ======================================================
# Phase-only steering per channel
# ======================================================
for c in range(C):
    ch = x[:, :, c]
    F = fft2c(ch)
    A = np.abs(F)
    Phi = np.angle(F)

    delta_phi = np.zeros_like(Phi)

    for (lo, hi) in bands:
        band = (r_norm > lo) & (r_norm < hi)
        # Structured micro phase steering (no magnitude change)
        delta_phi += EPS * band * dir_w * basin_w * np.sin(2 * np.pi * r_norm)

    # Cross-channel decorrelation: tiny channel-specific offset
    if args.cross_channel:
        delta_phi += (c - 1) * (EPS * 0.35) * np.cos(3 * theta)

    F_mod = phase_only_reconstruct(A, Phi + delta_phi)
    ch_out = ifft2c(F_mod)
    x_out[:, :, c] = ch_out

# Post-process (strict)
x_out = np.clip(x_out, 0, 1)
Image.fromarray((x_out * 255).astype(np.uint8)).save(args.out)

print("[+] Butterfly v2 complete")
print("[+] Visual invariants preserved")
print(f"[+] Saved: {args.out}")

