
# %%
# Let's load the original image

from PIL import Image
from IPython.display import display
import cv2
import numpy as np

# %%
# Let's define the filepaths
template_img_fp = "resources/Anwesenheitsliste_lt.png"
scanned_img_fp = "resources/KW_37_bsf-n135_p1.png"

# %%
# preprocessing
def preprocess(image, remove_coloured_pixels=True):
    """
    Convert to grayscale, blur, and perform Canny edge detection.
    """
    image = image.convert("RGB")
    image = np.array(image)

    saturation_threshold = 25
    if remove_coloured_pixels:
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

    return Image.fromarray(gray)
# %%
# Let's concentrate on the template first.
# we already got rid of the text that won't be present in the printed sheets
def get_keypoints(img_fp, display_keypoints=True):
    img = Image.open(img_fp)
    img_gray = preprocess(img)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(np.array(img_gray), None)
    img_keypoints = cv2.drawKeypoints(np.array(img_gray), keypoints, None, (0, 255, 0))
    
    if display_keypoints:
        display(Image.fromarray(img_keypoints))

    return keypoints, descriptors

anwesenheitsliste_lt_keypoints, anwesenheitsliste_lt_descriptors = get_keypoints(template_img_fp)
scanned_list_keypoints, scanned_list_descriptors = get_keypoints(scanned_img_fp)
# %%

# okay now we want to match the template onto the scanned image
def match_images(template_kp, template_desc, scanned_kp, scanned_desc, ratio_thresh=0.9, min_matches=10, display_matches=True):
    # Create FLANN matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Perform KNN matching
    matches = flann.knnMatch(template_desc, scanned_desc, k=2)

    print(f"len matches before filtering: {len(matches)}")

    # Apply Lowe's ratio test
    good_matches = []
    for m,n in matches:
        if m.distance < ratio_thresh * n.distance:
            good_matches.append(m)

    print(f"len matches after filtering: {len(good_matches)}")


    # Check if we have enough good matches
    if len(good_matches) < min_matches:
        print(f"Not enough matches found - {len(good_matches)}/{min_matches}")
        return None

    # Find homography
    dst_pts = np.float32([template_kp[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    src_pts = np.float32([scanned_kp[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # src is the scan, dst is the template
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matches_mask = mask.ravel().tolist()

    # Draw matches
    draw_params = dict(matchColor=(0,255,0),  # green matches
                       singlePointColor=None,
                       matchesMask=matches_mask,
                       flags=2)

    img_template = cv2.imread(template_img_fp, 0)
    img_scanned = cv2.imread(scanned_img_fp, 0)
    img_matches = cv2.drawMatches(img_template, template_kp, 
                                 img_scanned, scanned_kp, 
                                 good_matches, None, **draw_params)

    if display_matches:
        display(Image.fromarray(img_matches))

    return M, good_matches

# Usage
homography_matrix, good_matches = match_images(
    anwesenheitsliste_lt_keypoints,
    anwesenheitsliste_lt_descriptors,
    scanned_list_keypoints,
    scanned_list_descriptors
)

# %%
h_template, w_template = Image.open(template_img_fp).size

# Warp the scanned image
warped_scanned = cv2.warpPerspective(
    np.array(Image.open(scanned_img_fp)), 
    homography_matrix, 
    (w_template, h_template)
)

display(Image.fromarray(warped_scanned))
# %%
