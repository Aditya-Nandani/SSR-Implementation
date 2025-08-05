import numpy as np
import cv2
import os


def single_scale_retinex(img, sigma):
    """
    Apply single-scale Retinex to an image.
    """
    retinex = np.log10(img) - np.log10(cv2.GaussianBlur(img, (0, 0), sigma))
    return retinex


def normalize_channel(channel, low_percent=1, high_percent=99):
    """
    Normalize a single image channel based on percentiles.
    """
    low_val = np.percentile(channel, low_percent)
    high_val = np.percentile(channel, high_percent)
    channel = np.clip(channel, low_val, high_val)
    channel = (channel - np.min(channel)) / (np.max(channel) - np.min(channel)) * 255
    return channel


def SSR(img, sigma=30, clip_percent=1, apply_gamma=False, gamma=1.0):
    """
    Complete SSR pipeline with optional gamma correction.
    """
    img = np.float64(img) + 1.0  # Avoid log(0)
    retinex = single_scale_retinex(img, sigma)

    for i in range(3):  # RGB Channels
        retinex[:, :, i] = normalize_channel(retinex[:, :, i], clip_percent, 100 - clip_percent)

    if apply_gamma:
        retinex = np.power(retinex / 255.0, gamma) * 255

    return np.uint8(retinex)


def show_and_save(original, enhanced, save_path='SSR_output.jpg'):
    """
    Display and save the output.
    """
    cv2.imshow('Original Image', original)
    cv2.imshow('SSR Enhanced Image', enhanced)
    cv2.imwrite(save_path, enhanced)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    # Load image
    image_path = 'test.jpg'  # Replace with your own image
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB for correct color handling

    # Apply SSR
    sigma = 30
    gamma = 1.2
    enhanced_img = SSR(img, sigma=sigma, clip_percent=1, apply_gamma=True, gamma=gamma)

    # Convert back to BGR for saving/viewing in OpenCV
    enhanced_img_bgr = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    # Display and save
    show_and_save(img_bgr, enhanced_img_bgr, save_path='SSR_output.jpg')


if __name__ == "__main__":
    main()
