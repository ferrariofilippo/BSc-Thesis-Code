#!/usr/bin/env python3

import re
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
MESH_DIR   = Path("./meshes")
X_MIN, X_MAX = 1, 1260
# PATTERN    = re.compile(r"^(\d+)_Th_Iteration_(\d+)\.vtu$")
PATTERN    = re.compile(r"^(\d+)_Th_Iteration_0.vtu$")

# ── Parse files ───────────────────────────────────────────────────────────────
if not MESH_DIR.exists():
    raise FileNotFoundError(f"Directory '{MESH_DIR}' not found.")

x_values: list[int] = [ 0 for i in range(1, 1260 + 1)]

for path in MESH_DIR.iterdir():
    if not path.is_file():
        continue
    m = PATTERN.match(path.name)
    if m:
        x_values[int(m.group(1)) - 1] += 1

missing    = [i+1 for i in range(len(x_values)) if x_values[i] == 0]
duplicates = [i+1 for i in range(len(x_values)) if x_values[i] > 1]

print(f"\n Missing Meshes ({len(missing)}):")
if missing:
    for i in range(0, len(missing)):
        print(f"- {missing[i]}")
else:
    print("    (none — All the meshes were found")

print(f"\n  Duplicate Meshes ({len(duplicates)}):")
if duplicates:
    for i in range(0, len(duplicates)):
        print(f"- {duplicates[i]}")
else:
    print("    (none — no duplicate mesh)")

print()
