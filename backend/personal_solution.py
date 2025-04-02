# %%
import numpy as np
from pathlib import Path
from PIL import Image
from IPython.display import display
import cv2
# %%
scan_fp = Path("resources/KW_37_bsf-n135_p1.png")
template_fp = Path("resources/Anwesenheitsliste_lt.png")

scan = Image.open(scan_fp)
template = Image.open(template_fp)

# display(scan)
# display(template)
# %%
# preprocessing

def preprocess(image):
    """
    Convert to grayscale, blur, and perform Canny edge detection.
    """
    saturation_threshold = 25
    image = image.convert("RGB")
    image = np.array(image)

    # Convert image to HSV to inspect saturation.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Create a mask where the saturation is higher than the threshold.
    # Pixels with low saturation are near gray/black/white and are kept.
    colored_mask = hsv[:, :, 1] > saturation_threshold

    # Set colored pixels to white in the original image.
    # Note: colored_mask is a 2D boolean array. When used to index the 3-channel image,
    # it selects all channels for those pixels.
    image[colored_mask] = [255, 255, 255]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    median = cv2.medianBlur(gray, 5)
    edges = cv2.Canny(median, 50, 150, apertureSize=3)

    return edges


# %%
scan_preprocessed = preprocess(scan)
template_preprocessed = preprocess(template)

del scan, template

display(Image.fromarray(scan_preprocessed))
display(Image.fromarray(template_preprocessed))
# %%
def detect_lines(img):
    """
    Detect line segments using HoughLinesP.
    Adjust the thresholds, minLineLength, and maxLineGap as needed.
    """
    lines = cv2.HoughLinesP(img, 1, np.pi / 180, threshold=100,
                            minLineLength=50, maxLineGap=10)
    if lines is None:
        return []
    return lines[:, 0, :]  # each line: [x1, y1, x2, y2]

scan_lines = detect_lines(scan_preprocessed)
template_lines = detect_lines(template_preprocessed)
# %%
def visualize_lines(lines, img):
    """
    Draw lines on the image.
    """
    # If the image is grayscale (2D), convert it to BGR (3-channel) first.
    if len(img.shape) == 2 or img.shape[2] == 1:
        img_with_lines = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_with_lines = img.copy()

    for x1, y1, x2, y2 in lines:
        # Draw a red line (BGR: (0, 0, 255))
        cv2.line(img_with_lines, (x1, y1), (x2, y2), (0, 0, 255), 2)

    display(Image.fromarray(img_with_lines))

visualize_lines(scan_lines, np.array(scan_preprocessed))
visualize_lines(template_lines, np.array(template_preprocessed))

print(f"amt of scan lines: {len(scan_lines)}")
print(f"amt of template lines: {len(template_lines)}")
# %%
