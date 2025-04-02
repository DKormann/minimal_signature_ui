# %%
from pathlib import Path
from PIL import Image
import cv2
import numpy as np
from IPython.display import display
import matplotlib.pyplot as plt
# %%
rotated_scan_fp = Path('resources/rotated/rotated_signature_scan.png')
# markers to identify the corners of the rotated scan
bottom_left_fp = Path('resources/rotated/bottom_left.png')
top_right_fp = Path('resources/rotated/top_right.png')

scan = cv2.imread(rotated_scan_fp)
bottom_left_logo = cv2.imread(bottom_left_fp)
top_right_logo = cv2.imread(top_right_fp)
# %%
def preprocess_image(img):
    return img

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray

scan_gray = preprocess_image(scan)
display(Image.fromarray(scan_gray))

bottom_left_gray = preprocess_image(bottom_left_logo)
top_right_gray = preprocess_image(top_right_logo)

display(Image.fromarray(bottom_left_gray))
display(Image.fromarray(top_right_gray))
# %%
def find_logos(image, logo):
    result = cv2.matchTemplate(image, logo, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(f"max_val: {max_val}")
    print(f"max_loc: {max_loc}")
    print(f"min_val: {min_val}")
    print(f"min_loc: {min_loc}")
    print("----")

    return max_loc

bottom_left_loc = find_logos(scan_gray, bottom_left_gray)
top_right_loc = find_logos(scan_gray, top_right_gray)

# %%
def visualize_loc(image, logo, loc):
    h, w = logo.shape[0], logo.shape[1]
    cv2.rectangle(image, loc, (loc[0] + w, loc[1] + h), (0, 0, 255), 2)
    display(Image.fromarray(image))

visualize_loc(scan.copy(), bottom_left_gray, bottom_left_loc)

# %%
visualize_loc(scan.copy(), top_right_gray, top_right_loc)
print(top_right_loc)
# %%
def crop_image(image, loc_bottom_left, logo_bottom_left, loc_top_right, logo_top_right):
    h_bottom_left, w_bottom_left = logo_bottom_left.shape[0], logo_bottom_left.shape[1]
    print(f"h_bottom_left {h_bottom_left}")
    print(f"w_bottom_left {w_bottom_left}")
    h_top_right, w_top_right = logo_top_right.shape[0], logo_top_right.shape[1]

    print(f"loc_bottom_left: {loc_bottom_left}")
    y_bottom_left = loc_bottom_left[1] + h_bottom_left
    print(f"y_bottom_left: {y_bottom_left}")

    x_bottom_left = loc_bottom_left[0]
    print(f"x_bottom_left: {x_bottom_left}")

    print(f"loc_top_right: {loc_top_right}")
    x_top_right = loc_top_right[0] + w_top_right
    print(f"x_top_right: {x_top_right}")

    y_top_right = loc_top_right[1]
    print(f"y_top_right: {y_top_right}")

    cropped_scan = image[y_top_right:y_bottom_left, x_bottom_left:x_top_right]
    return cropped_scan

cropped_scan = crop_image(scan, bottom_left_loc, bottom_left_gray, top_right_loc, top_right_gray)
display(Image.fromarray(cropped_scan))
# %%
template_fp = Path('resources/Anwesenheitsliste_lt.png')
template = cv2.imread(template_fp)

print(cropped_scan.shape)
print(f"template shape: {template.shape}")
# %%
def rescale_image(image, template):
    h, w = template.shape[0], template.shape[1]
    return cv2.resize(image, (w, h))

rescaled_cropped_scan = rescale_image(cropped_scan, template)
display(Image.fromarray(rescaled_cropped_scan))
# %%
def plot_images(images):
    fig, ax = plt.subplots(1, len(images), figsize=(10, 5))

    for i, image in enumerate(images):
        ax[i].imshow(image)

    plt.show()
# %%
original_scan_fp = Path('resources/KW_37_bsf-n135_p1.png')
original_scan = cv2.imread(original_scan_fp)
plot_images([original_scan, rescaled_cropped_scan, template])
# %%
