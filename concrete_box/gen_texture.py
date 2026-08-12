import numpy as np
import cv2
import os

output_dir = r"C:\Users\Hema Pandey\PycharmProjects\3Dgame\concrete_box"
output_path = os.path.join(output_dir, "concrete.png")

np.random.seed(42)
size = 512

# --- Base grey concrete color ---
base = np.full((size, size, 3), (130, 128, 125), dtype=np.float32)

# --- Large-scale variation (simulate concrete pour lines) ---
noise_large = np.zeros((size, size), dtype=np.float32)
for scale in [64, 128, 256]:
    small = np.random.rand(size // scale + 2, size // scale + 2).astype(np.float32)
    large = cv2.resize(small, (size, size), interpolation=cv2.INTER_CUBIC)
    noise_large += large * scale / 64

noise_large = cv2.normalize(noise_large, None, 0, 1, cv2.NORM_MINMAX)
variation = (noise_large - 0.5) * 30  # +/- 15 brightness shift
for c in range(3):
    base[:, :, c] += variation

# --- Medium grain noise ---
grain = np.random.randn(size, size).astype(np.float32) * 8
for c in range(3):
    base[:, :, c] += grain

# --- Subtle color tint variation (warm/cool patches) ---
tint_r = cv2.resize(np.random.rand(8, 8).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
tint_b = cv2.resize(np.random.rand(8, 8).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
base[:, :, 0] += (tint_r - 0.5) * 10  # red channel
base[:, :, 2] += (tint_b - 0.5) * 8   # blue channel

# --- Cracks ---
def draw_crack(img, start, angle, length, thickness=1, darkness=40):
    x, y = float(start[0]), float(start[1])
    rad = np.deg2rad(angle)
    dx = np.cos(rad)
    dy = np.sin(rad)
    for i in range(length):
        xi, yi = int(x), int(y)
        if 0 <= xi < size and 0 <= yi < size:
            for c in range(3):
                img[yi, xi, c] = max(0, img[yi, xi, c] - darkness * (1 - i / length * 0.5))
        # slight random walk
        angle += np.random.uniform(-5, 5)
        rad = np.deg2rad(angle)
        dx = np.cos(rad)
        dy = np.sin(rad)
        x += dx
        y += dy

rng = np.random.RandomState(7)
for _ in range(6):
    sx = rng.randint(50, size - 50)
    sy = rng.randint(50, size - 50)
    angle = rng.uniform(0, 360)
    length = rng.randint(40, 120)
    draw_crack(base, (sx, sy), angle, length, darkness=rng.randint(25, 50))

# --- Aggregate (pebble) spots ---
for _ in range(200):
    cx = rng.randint(0, size)
    cy = rng.randint(0, size)
    r = rng.randint(2, 6)
    brightness = rng.uniform(-15, 15)
    color = (
        float(base[cy, cx, 0]) + brightness,
        float(base[cy, cx, 1]) + brightness,
        float(base[cy, cx, 2]) + brightness,
    )
    cv2.circle(base, (cx, cy), r, color, -1)

# --- Clamp and convert ---
base = np.clip(base, 60, 200).astype(np.uint8)

# --- Light blur to smooth out harsh edges ---
base = cv2.GaussianBlur(base, (3, 3), 0.8)

cv2.imwrite(output_path, base)
print(f"Saved: {output_path}")
print(f"Shape: {base.shape}, dtype: {base.dtype}")
