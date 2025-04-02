# %%
import cv2
import numpy as np
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

# %%
scan_fp = "resources/KW_37_bsf-n135_p1.png"
template_fp = "resources/Anwesenheitsliste_lt.png"

# scan = np.array(Image.open(scan_fp))
# template = np.array(Image.open(template_fp))
# %%
sift = cv2.SIFT_create()

def filter_matches(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    return good_matches

def find_matches(scan, template):
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(scan, None)
    kp2, des2 = sift.detectAndCompute(template, None)

    # ISSUE! THE DESCRIPTORS DESCRIBE THE AREAS AROUND THE KEYPOINTS.
    # SINCE A GRID IS OBVIOUSLY REPEATING, THESE DESCRIPTORS WILL BE VERY SIMILAR.

    # Use FLANN based matcher
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=100)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    print(f"all matches: {len(matches)}")

    # Apply Lowe's ratio test to find good matches
    good_matches = filter_matches(matches)

    # Draw matches
    result_img = cv2.drawMatches(scan, kp1, template, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    display(Image.fromarray(result_img))

    return good_matches, kp1, kp2

# good_matches, kpscan, kptemplate = find_matches(scan, template)

# %%
# let's visualize the keypoints
def visualize_keypoints(img, kp):
    img = cv2.drawKeypoints(img, kp, None)
    display(Image.fromarray(img))

# visualize_keypoints(scan, kpscan)
# visualize_keypoints(template, kptemplate)
# %%
# let's compute the transformation matrix
def trans_matrix(matches, kpscan, kptemplate):
    # We need at least 4 matches to compute homography
    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return None, None

    # Extract the coordinates of the matched keypoints.
    src_pts = np.float32([kpscan[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kptemplate[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    # Compute the inliers using RANSAC.
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M, inliers

# M, inliers = trans_matrix(good_matches, kpscan, kptemplate)

# %%
def project_image(scan_img, templ_img, M):
    projected_image = cv2.warpAffine(src=scan_img, M=M, dsize=(templ_img.shape[1], templ_img.shape[0]), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    
    print(f"shape of projected image: {projected_image.shape}")
    print(f"shape of template image: {templ_img.shape}")
    return projected_image

# projected_image = project_image(scan_img=scan, templ_img=template, M=M)
# images = [np.array(scan), np.array(template), np.array(projected_image)]

# titles = ["scan", "template", "projected scan"]

# create_image_grid(images, titles)
# %%
# let's create a pipeline

def sift_anwesenheitsliste(from_img_fp, to_img_fp):
    scan = np.array(Image.open(from_img_fp))
    template = np.array(Image.open(to_img_fp))

    good_matches, kpscan, kptemplate = find_matches(scan, template)
    M, inliers = trans_matrix(good_matches, kpscan, kptemplate)

    projected_image = project_image(scan_img=scan, templ_img=template, M=M)

    return M, inliers, projected_image

M, inliers, projected_image_array = sift_anwesenheitsliste(scan_fp, template_fp)
projected_image = Image.fromarray(projected_image_array)
display(projected_image)
display(Image.open(template_fp))

# %%
# let's save the projected image
rotated_signatures_fp = "resources/rotated/rotated_signature_scan.png"
projected_image.save(rotated_signatures_fp)

