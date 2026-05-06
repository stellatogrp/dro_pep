import numpy as np
import matplotlib.pyplot as plt
from skimage import data, transform, util

TARGET = 128

# Load standard grayscale test images
images = {
    "camera":   data.camera(),
    "moon":     data.moon(),
    "coins":    data.coins(),
    "text":     data.text(),
    "clock":    data.clock(),
    "page":     data.page(),
    "cell":     data.cell(),
}

# Downscale to 64x64 (anti-aliased, returns float in [0, 1])
target = (TARGET, TARGET)
small = {
    name: transform.resize(img, target, anti_aliasing=True, preserve_range=False)
    for name, img in images.items()
}

# Quick sanity check
for name, img in small.items():
    print(f"{name:8s} shape={img.shape} dtype={img.dtype} range=[{img.min():.3f}, {img.max():.3f}]")

# Plot grid
fig, axes = plt.subplots(2, 4, figsize=(10, 5))
for ax, (name, img) in zip(axes.ravel(), small.items()):
    ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(name)
    ax.axis("off")
for ax in axes.ravel()[len(small):]:
    ax.axis("off")
plt.tight_layout()
plt.show()

# Optionally save as .npy or .png
np.save(f"test_images_{TARGET}.npy", np.stack(list(small.values())))
# or save individual PNGs:
# from skimage.io import imsave
# for name, img in small.items():
#     imsave(f"{name}_64.png", util.img_as_ubyte(img))
