"""
chip_generator_profiles.py

Generates 2D steady-state chip-style heat maps + sensor readings
for several processor classes (laptop + mobile). Presets (die mm^2, TDP)
are taken from public product briefs/spec pages (see citations in comments).

Run in Colab or locally:
pip install numpy scipy matplotlib
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
import os
from math import ceil

# --------------------------
# Processor profiles (sourced/approximated)
# Notes:
# - die_area_mm2: approximate die area used to scale "grid" size.
# - tdp_w: typical package TDP or sustained power guidance (approx).
# - hotspot_strategy: rules for placing hotspots
# Sources (examples): Intel/AMD product pages, Qualcomm/Apple/MediaTek briefs.
# See assistant citations in chat for exact sources used to choose these numbers.
# --------------------------

processor_profiles = {
    # LAPTOPS
    "intel_core_mobile": {
        "label": "Intel Core (mobile i3/i5/i7 family)",
        "die_area_mm2": 200,      # approximate package die area (representative)
        "tdp_w": 45.0,            # mobile high-power SKU (W)
        "num_hotspots": 6,
        "hotspot_shape": "rect",  # rect blocks for cores
        "comments": "Representative mobile Core family (high-end mobile i7). See Intel ARK pages."
    },
    "amd_ryzen_mobile": {
        "label": "AMD Ryzen (mobile)",
        "die_area_mm2": 200,      # use combined CCD+IOD representation; adjust if needed
        "tdp_w": 45.0,
        "num_hotspots": 6,
        "hotspot_shape": "rect",
        "comments": "Based on AMD mobile product pages (CCD+IOD sizes)."
    },
    "apple_silicon": {
        "label": "Apple Silicon (M-series)",
        "die_area_mm2": 150,
        "tdp_w": 20.0,            # Apple laptops run lower sustained power
        "num_hotspots": 4,
        "hotspot_shape": "mixed", # core blocks + GPU area
        "comments": "SoC style; fewer large blocks; more integrated."
    },
    "intel_xeon_mobile": {
        "label": "Intel Xeon (mobile/workstation)",
        "die_area_mm2": 300,
        "tdp_w": 85.0,
        "num_hotspots": 8,
        "hotspot_shape": "rect",
        "comments": "Workstation-class, higher TDP."
    },
    "amd_threadripper": {
        "label": "AMD Threadripper (workstation)",
        "die_area_mm2": 900,
        "tdp_w": 350.0,
        "num_hotspots": 16,
        "hotspot_shape": "rect",
        "comments": "High core-count, many hotspots (approx)."
    },

    # MOBILES
    "snapdragon_flagship": {
        "label": "Qualcomm Snapdragon (flagship)",
        "die_area_mm2": 100,
        "tdp_w": 5.0,             # sustained power envelope (approx)
        "num_hotspots": 3,
        "hotspot_shape": "small_rect",
        "comments": "Flagship mobile SoC sustained power ~4-6W (product briefs)."
    },
    "apple_a_series": {
        "label": "Apple A-series (iPhone SoC)",
        "die_area_mm2": 120,
        "tdp_w": 5.0,
        "num_hotspots": 3,
        "hotspot_shape": "mixed",
        "comments": "Mobile SoC (A-series) typical few-watt sustained power."
    },
    "mediatek_dimensity": {
        "label": "MediaTek Dimensity (flagship)",
        "die_area_mm2": 100,
        "tdp_w": 5.0,
        "num_hotspots": 3,
        "hotspot_shape": "small_rect",
        "comments": "Flagship Dimensity family; similar mobile envelope."
    }
}

# --------------------------
# Utility: map die area to grid size
# We choose a base grid scale: ~1 mm^2 -> ~6x6 grid units (empirical)
# so grid_size = ceil(sqrt(die_area_mm2) * scale_factor)
# --------------------------
SCALE_FACTOR = 6.0  # tuneable: higher -> larger grid resolution for given die area

def die_area_to_grid(die_mm2):
    side_mm = np.sqrt(die_mm2)
    grid_side = max(32, int(ceil(side_mm * SCALE_FACTOR)))  # min 32
    return grid_side

# --------------------------
# Create 2D Laplacian (Dirichlet interior)
# --------------------------
def create_laplacian(N):
    size = N * N
    main_diag = -4 * np.ones(size)
    side_diag = np.ones(size)
    diagonals = [main_diag, side_diag, side_diag, side_diag, side_diag]
    offsets = [0, -1, 1, -N, N]
    L = sp.diags(diagonals, offsets, shape=(size, size), format='csr')
    # Fix boundary rows where left-right neighbor crosses a row (remove wrap)
    # We'll keep it simple: users can refine boundary treatment later.
    return L

# --------------------------
# Hotspot placement helpers
# --------------------------
def place_rect_hotspot(Q, cx, cy, w, h, power):
    x0 = max(0, int(cx - w//2))
    x1 = min(Q.shape[0], int(cx + w//2))
    y0 = max(0, int(cy - h//2))
    y1 = min(Q.shape[1], int(cy + h//2))
    Q[x0:x1, y0:y1] += power

def place_gaussian_hotspot(Q, cx, cy, sigma, intensity):
    x = np.arange(Q.shape[0])
    y = np.arange(Q.shape[1])
    X, Y = np.meshgrid(x, y, indexing='ij')
    dist2 = (X-cx)**2 + (Y-cy)**2
    Q += intensity * np.exp(-dist2 / (2*sigma*sigma))

# --------------------------
# Main generator for a single processor profile
# --------------------------
def generate_sample_for_profile(profile_key, idx, out_dir, sensor_layout="fixed"):
    profile = processor_profiles[profile_key]
    gridN = die_area_to_grid(profile["die_area_mm2"])
    N = gridN
    L = create_laplacian(N)

    # total power to distribute approximated from TDP (scale factor to approximate steady-state Q)
    total_power = profile["tdp_w"]

    # Initialize Q (power density per node)
    Q = np.zeros((N, N))

    # Heuristic: distribute total_power among hotspots; each hotspot contributes part of Q (W)
    num_hs = profile["num_hotspots"]
    # fraction of TDP assigned to hotspots (rest to background leakage): e.g., 0.9
    hs_power_fraction = 0.9
    per_hs_power = (total_power * hs_power_fraction) / max(1, num_hs)

    # Place hotspots according to hotspot_shape preference
    rng = np.random.RandomState(idx + 12345)  # deterministic-ish by idx
    if profile["hotspot_shape"] in ["rect", "small_rect"]:
        # place rectangular blocks roughly across the die
        for h in range(num_hs):
            cx = rng.randint(int(0.15*N), int(0.85*N))
            cy = rng.randint(int(0.15*N), int(0.85*N))
            # size scales with die size and hotspot type
            if profile["hotspot_shape"] == "small_rect":
                w = max(3, int(N * 0.06))
                hgt = max(3, int(N * 0.06))
            else:
                w = max(6, int(N * 0.12))
                hgt = max(6, int(N * 0.12))
            place_rect_hotspot(Q, cx, cy, w, hgt, per_hs_power)
    elif profile["hotspot_shape"] == "mixed":
        # some gaussian + one larger block simulating GPU
        for h in range(num_hs):
            if h == 0:
                # larger gaussian center-ish
                cx = int(N*0.5 + rng.randint(-int(N*0.1), int(N*0.1)))
                cy = int(N*0.5 + rng.randint(-int(N*0.1), int(N*0.1)))
                sigma = max(6, int(N*0.12))
                place_gaussian_hotspot(Q, cx, cy, sigma, per_hs_power*1.5)
            else:
                cx = rng.randint(int(0.2*N), int(0.8*N))
                cy = rng.randint(int(0.2*N), int(0.8*N))
                sigma = max(3, int(N*0.06))
                place_gaussian_hotspot(Q, cx, cy, sigma, per_hs_power)
    else:
        # fallback gaussian hotspots
        for h in range(num_hs):
            cx = rng.randint(int(0.2*N), int(0.8*N))
            cy = rng.randint(int(0.2*N), int(0.8*N))
            sigma = max(4, int(N*0.08))
            place_gaussian_hotspot(Q, cx, cy, sigma, per_hs_power)

    # Add a global gradient (simulate uneven cooling / heatspreader effect)
    grad = np.linspace(0.0, 0.2*total_power, N)
    gradient_field = np.tile(grad, (N,1))
    Q += gradient_field * 0.01  # small background load scaling

    # Solve discrete Laplacian: ∇^2 T + Q = 0  -> L * T_flat = -Q_flat
    b = -Q.flatten()
    # Use a sparse solver
    try:
        T_flat = spla.spsolve(L, b)
    except Exception as e:
        # fallback to pseudo-inverse if solver fails (shouldn't normally)
        T_flat = np.linalg.lstsq(L.toarray(), b, rcond=None)[0]
    T = T_flat.reshape((N, N))

    # Normalize and scale into plausible temperature range
    T = T - T.min()
    T = 30.0 + (T / (T.max() + 1e-12)) * 70.0  # 30°C -> 100°C range as plausible

    # Add slight Gaussian measurement noise
    noise = np.random.normal(0, 0.3, (N, N))
    T += noise

    # Sensor layout: fixed grid or random; for chip realism we supply a fixed layout (center + corners + midlines)
    if sensor_layout == "fixed":
        # choose up to 16 sensors at fixed relative positions on this grid
        coords = [
            (int(0.2*N), int(0.2*N)), (int(0.2*N), int(0.5*N)), (int(0.2*N), int(0.8*N)),
            (int(0.5*N), int(0.2*N)), (int(0.5*N), int(0.5*N)), (int(0.5*N), int(0.8*N)),
            (int(0.8*N), int(0.2*N)), (int(0.8*N), int(0.5*N)), (int(0.8*N), int(0.8*N)),
            (int(0.1*N), int(0.5*N)), (int(0.9*N), int(0.5*N)),
            (int(0.5*N), int(0.1*N)), (int(0.5*N), int(0.9*N)),
            (int(0.3*N), int(0.3*N)), (int(0.7*N), int(0.7*N)), (int(0.5*N), int(0.3*N))
        ]
        coords = coords[:16]
    else:
        k = 16
        rng2 = np.random.RandomState(idx + 999)
        coords = [(int(rng2.randint(0,N)), int(rng2.randint(0,N))) for _ in range(k)]

    sensor_values = np.array([T[r,c] for r,c in coords])

    # Save outputs
    # directories
    proc_dir = os.path.join(out_dir, profile_key)
    os.makedirs(proc_dir, exist_ok=True)
    img_dir = os.path.join(proc_dir, "images")
    arr_dir = os.path.join(proc_dir, "arrays")
    sen_dir = os.path.join(proc_dir, "sensors")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(arr_dir, exist_ok=True)
    os.makedirs(sen_dir, exist_ok=True)

    # Save numpy array and image
    np.save(os.path.join(arr_dir, f"{profile_key}_{idx}.npy"), T)
    # ---- FORMAT OUTPUT LIKE SAMPLE IMAGE ----

# Normalize cleanly to 0–1 for rendering
    T_render = T - T.min()
    T_render = T_render / (T_render.max() + 1e-12)

# Add subtle diagonal cooling gradient (for that smooth sweep look)
    grad = (np.linspace(0, 1, N)[:,None] + np.linspace(0, 1, N)[None,:]) / 2

    T_render = 0.85 * T_render + 0.15 * grad

# Add very light speckle noise (thermal camera feel)
    T_render += np.random.normal(0, 0.01, (N, N))

    plt.figure(figsize=(4,4))
    plt.imshow(T_render,
           cmap='jet',              # match your image
           interpolation='bicubic') # smooth blending

    plt.axis('off')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)

    plt.savefig(os.path.join(img_dir, f"{profile_key}_{idx}.png"),
            dpi=200,
            bbox_inches='tight',
            pad_inches=0)

    plt.close()

    np.savez(os.path.join(sen_dir, f"{profile_key}_{idx}.npz"),
             coords=coords, values=sensor_values, profile=profile_key)

    return True

# --------------------------
# Batch runner
# --------------------------
def generate_dataset(out_dir="generated_chips", samples_per_profile=500):
    for key in processor_profiles.keys():
        print("Generating for profile:", key)
        for i in range(samples_per_profile):
            generate_sample_for_profile(key, i, out_dir, sensor_layout="fixed")
    print("Done.")

# --------------------------
# If run as script, generate a small dataset (change samples_per_profile as needed)
# --------------------------
if __name__ == "__main__":
    # Example: 200 samples per profile -> 8 laptop + 3 mobile = 8*200 = 1600 images
    generate_dataset(out_dir="generated_chips", samples_per_profile=1)
