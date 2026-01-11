from PIL import Image
import numpy as np
from scipy.signal import convolve2d

def load_image(path):
    img = Image.open(path)

    img_np = np.array(img)

    if img_np.ndim == 2:
        unique_vals = np.unique(img_np)
        if set(unique_vals).issubset({0, 1, 255}):
            return img_np > 0
        return img_np.astype(np.uint8)

    if img_np.ndim == 3:
        return img_np.astype(np.uint8)

    raise ValueError("Unsupported image format")

def edge_detection(image):
    if image.ndim == 3:
        gray = image.mean(axis=2)
    else:
        gray = image

    gray = gray.astype(float)

    kernelY = np.array([
        [ 1,  2,  1],
        [ 0,  0,  0],
        [-1, -2, -1]
    ])

    kernelX = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ])

    edgeY = np.convolve2d(gray, kernelY, mode="same", boundary="fill", fillvalue=0)
    edgeX = np.convolve2d(gray, kernelX, mode="same", boundary="fill", fillvalue=0)

    edgeMAG = np.sqrt(edgeX**2 + edgeY**2)
    return edgeMAG
