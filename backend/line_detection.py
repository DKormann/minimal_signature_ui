# %%
import numpy as np
from pathlib import Path
from PIL import Image
from IPython.display import display
import cv2
import math
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
                            minLineLength=250, maxLineGap=10)
    if lines is None:
        return []


    result = lines[:, 0, :]  #' each line: [x1, y1, x2, y2]
    lenght = [(((x1 - x2)**2 + (y1 - y2)**2)**0.5, (x1,y1, x2,y2))for x1,y1,x2,y2 in result]
    sortedlenght = list(reversed(sorted(lenght)))
    biggest_lines = sortedlenght[:10]
    angles = [round(math.degrees(math.atan2((x1 - x2), (y1 - y2)) + math.pi), 2) for lenght, (x1, y1, x2, y2) in biggest_lines]
    

    print (biggest_lines, angles)
    return [lines_coordinates for lenght, lines_coordinates in biggest_lines]

scan_lines = detect_lines(scan_preprocessed)
# print(scan_lines)

#%%
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
# I forgot what was this
import math

math.atan2(-1,0)
# %%
def separate_lines(lines, angle_threshold=45):
    # if the angle is <= 45° the line is considered horrizontal
    horizontal_lines = []
    vertical_lines = []
    angles_deg = []

    for (x1, y1, x2, y2) in lines:
        dx = x2 - x1
        dy = y2 - y1
        angle = math.degrees(math.atan2(dy, dx))

        angle = angle % 180  
        if angle > 90:
            angle -= 180     

        angles_deg.append(angle)

        if abs(angle) <= angle_threshold:
            horizontal_lines.append((x1, y1, x2, y2))
        else:
            vertical_lines.append((x1, y1, x2, y2))

    return horizontal_lines, vertical_lines, angles_deg

horrizontal_lines, vertical_lines, all_angles = separate_lines(scan_lines, angle_threshold=45)
print(f"number of horrizontal lines: {len(horrizontal_lines)}")
print(f"number of vertical lines: {len(vertical_lines)}")

# %%