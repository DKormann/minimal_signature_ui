import cv2
import numpy as np
from PIL import Image

def crop_to_largest_contour(pil_image, margin=10, morph_kernel_size=5):
    # image -> np image -> bin image
    arr = np.array(pil_image.convert('RGB'))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # with otsu the threshold is calculated automaticly
    kernel = np.ones((morph_kernel_size, morph_kernel_size), np.uint8)
    closed = cv2.morphologyEx(255 - bin_img, cv2.MORPH_CLOSE, kernel, iterations=1)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return pil_image
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    H, W = arr.shape[:2]
    return pil_image.crop((
        max(0, x - margin), 
        max(0, y - margin),
        min(W, x + w + margin),
        min(H, y + h + margin)
    ))

cropped = crop_to_largest_contour(rotated_image, margin=10, morph_kernel_size=5)
display(cropped)