#!/usr/bin/env python3
"""
Side-by-side map viewer for debugging the path planner.

Left:  original map image
Right: binary occupancy grid (with dilation toggle)

Hover over either panel to see (u, v) pixel coords and (x, y) world coords.

Usage:
    python3 scripts/visualize_map.py
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons
from pathlib import Path
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
MAP_DIR = SCRIPT_DIR.parent / "maps"
MAP_YAML = MAP_DIR / "stata_basement.yaml"
MAP_PNG = MAP_DIR / "stata_basement.png"

DILATION_RADIUS = 10  # pixels

with open(MAP_YAML) as f:
    meta = yaml.safe_load(f)

resolution = meta["resolution"]
origin_x, origin_y, origin_yaw = meta["origin"]
occupied_thresh = meta["occupied_thresh"]
free_thresh = meta["free_thresh"]
negate = meta["negate"]

cos_yaw = np.cos(origin_yaw)
sin_yaw = np.sin(origin_yaw)

raw_img = cv2.imread(str(MAP_PNG), cv2.IMREAD_GRAYSCALE)
if raw_img is None:
    raise FileNotFoundError(f"Could not load {MAP_PNG}")

height, width = raw_img.shape

# map_server flips vertically so row 0 = bottom of image
flipped = np.flipud(raw_img).astype(np.float64)

if negate:
    occ = flipped / 255.0
else:
    occ = (255.0 - flipped) / 255.0

# 1 = occupied/unknown, 0 = free
occupied = np.zeros_like(occ, dtype=np.uint8)
occupied[occ > occupied_thresh] = 1
occupied[(occ >= free_thresh) & (occ <= occupied_thresh)] = 1

kernel_size = 2 * DILATION_RADIUS + 1
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
dilated = cv2.dilate(occupied, kernel, iterations=1)

def grid_to_world(u, v):
    """Grid pixel (u, v) → world (x, y)."""
    mx = u * resolution
    my = v * resolution
    x = origin_x + mx * cos_yaw - my * sin_yaw
    y = origin_y + mx * sin_yaw + my * cos_yaw
    return x, y

fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(16, 7))
fig.subplots_adjust(bottom=0.08, left=0.05, right=0.95)

ax_left.imshow(flipped, cmap="gray", origin="lower")
ax_left.set_title("Original Map")
ax_left.set_xlabel("u (pixels)")
ax_left.set_ylabel("v (pixels)")

grid_img = ax_right.imshow(occupied, cmap="gray_r", origin="lower",
                           vmin=0, vmax=1)
ax_right.set_title("Occupancy Grid")
ax_right.set_xlabel("u (pixels)")
ax_right.set_ylabel("v (pixels)")

check_ax = fig.add_axes([0.02, 0.01, 0.12, 0.06])
check = CheckButtons(check_ax, ["Dilated"], [False])

def toggle_dilation(label):
    active = check.get_status()[0]
    grid_img.set_data(dilated if active else occupied)
    ax_right.set_title("Occupancy Grid (dilated)" if active else "Occupancy Grid")
    fig.canvas.draw_idle()

check.on_clicked(toggle_dilation)

def make_format_coord(ax):
    def format_coord(x, y):
        u, v = int(round(x)), int(round(y))
        if 0 <= u < width and 0 <= v < height:
            wx, wy = grid_to_world(u, v)
            return f"pixel=({u}, {v})  world=({wx:.2f}, {wy:.2f})"
        return f"pixel=({u}, {v})  [out of bounds]"
    return format_coord

ax_left.format_coord = make_format_coord(ax_left)
ax_right.format_coord = make_format_coord(ax_right)

plt.suptitle(
    f"stata_basement — {width}x{height} px, "
    f"res={resolution} m/px, "
    f"origin=({origin_x}, {origin_y}, yaw={origin_yaw:.2f})",
    fontsize=11,
)
plt.show()
