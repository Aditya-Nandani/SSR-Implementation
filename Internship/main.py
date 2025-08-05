import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_surround(image, sigma):
    return cv2.GaussianBlur(image, (0, 0), sigma)

def single_scale_retinex(image, sigma):
    image = image.astype(np.float32) + 1.0  # Avoid log(0)
    gaussian_blur = gaussian_surround(image, sigma)
    retinex = np.log(image) - np.log(gaussian_blur + 1.0)
    return retinex

def apply_ssr(image, sigma=30):
    result = np.zeros_like(image, dtype=np.float32)
    for i in range(3):  # Assuming RGB image
        result[:, :, i] = single_scale_retinex(image[:, :, i], sigma)
    # Normalize to 0-255 for display
    result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(result)

img = cv2.imread('test.jpg')  # Replace with your frame from vehicle
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

enhanced_img = apply_ssr(img, sigma=30)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(enhanced_img)
plt.title("SSR Enhanced")
plt.axis("off")
plt.show()
