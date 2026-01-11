import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from skimage.filters import median
from skimage.morphology import ball

from image_utils import load_image, edge_detection


# --------------------------------------------------# 
image_path = "my_image.jpg"  
color_image = load_image(image_path)

plt.imshow(color_image)
plt.title("Original image")
plt.axis("off")
plt.show()


# --------------------------------------------------# 
clean_image = median(color_image, ball(3))

plt.imshow(clean_image.astype(np.uint8))
plt.title("After median filtering")
plt.axis("off")
plt.show()


# --------------------------------------------------# 
edgeMAG = edge_detection(clean_image)


# --------------------------------------------------# 
plt.figure(figsize=(6, 4))
plt.hist(edgeMAG.flatten(), bins=100)
plt.title("Histogram of edgeMAG values")
plt.xlabel("Magnitude")
plt.ylabel("Frequency")
plt.show()

threshold = 100
edge_binary = edgeMAG > threshold


# --------------------------------------------------# 
plt.figure(figsize=(6, 6))
plt.imshow(edge_binary, cmap="gray")
plt.title("Binary Edge Image")
plt.axis("off")
plt.show()

edge_image = Image.fromarray((edge_binary * 255).astype(np.uint8))
edge_image.save("my_edges.png")
