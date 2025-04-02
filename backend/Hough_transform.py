# %%
# Let's load the original image

from PIL import Image
from IPython.display import display
import cv2
import math
import numpy as np

# %%
# Let's define the filepaths
template_img_fp = "resources/Anwesenheitsliste_lt.png"
scanned_img_fp = "resources/KW_37_bsf-n135_p1.png"

# %%
def preprocess(image, remove_coloured_pixels=True):
    """
    Convert to grayscale, blur, and perform Canny edge detection.
    """
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
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150, apertureSize=3)
    return edges

scan = Image.open(scanned_img_fp)
display(scan)

display(Image.fromarray(preprocess(np.array(scan))))
# %%

def detect_lines(edges):
    """
    Detect line segments using HoughLinesP.
    Adjust the thresholds, minLineLength, and maxLineGap as needed.
    """
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100,
                            minLineLength=50, maxLineGap=10)
    if lines is None:
        return []
    return lines[:, 0, :]  # each line: [x1, y1, x2, y2]

def line_angle(line):
    """
    Compute the angle (in degrees) of a line segment.
    """
    x1, y1, x2, y2 = line
    angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
    if angle < 0:
        angle += 180  # normalize to [0, 180)
    return angle

def cluster_lines(lines, angle_threshold=10):
    """
    Cluster lines into two groups based on their orientation.
    This simplistic approach compares each line’s angle to the average.
    """
    group1, group2 = [], []
    for line in lines:
        angle = line_angle(line)
        # If group1 is empty, add the first line
        if not group1:
            group1.append(line)
        else:
            avg_angle1 = np.mean([line_angle(l) for l in group1])
            if abs(angle - avg_angle1) < angle_threshold:
                group1.append(line)
            else:
                group2.append(line)
    return group1, group2

def compute_intersection(line1, line2):
    """
    Compute the intersection point of two lines given in (x1,y1,x2,y2) format.
    Returns None if lines are parallel.
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    # Represent lines in ax + by = c form.
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = a1 * x1 + b1 * y1

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = a2 * x3 + b2 * y3

    determinant = a1 * b2 - a2 * b1
    if abs(determinant) < 1e-10:
        return None  # Lines are parallel.
    x = (b2 * c1 - b1 * c2) / determinant
    y = (a1 * c2 - a2 * c1) / determinant
    return [x, y]

def compute_intersections(lines_group1, lines_group2):
    """
    Compute intersections between every line in group1 and every line in group2.
    """
    intersections = []
    for line1 in lines_group1:
        for line2 in lines_group2:
            pt = compute_intersection(line1, line2)
            if pt is not None:
                intersections.append(pt)
    return np.array(intersections)

def sort_grid_points(points):
    """
    A simple method to sort grid points.
    This sorts points by their y coordinate first, then by x.
    For a rotated grid, you might consider a more robust clustering.
    """
    sorted_points = sorted(points, key=lambda p: (p[1], p[0]))
    return np.array(sorted_points)

# %%
# Load the documents.
imgA = cv2.imread(template_img_fp)  # Grid-only document
imgB = cv2.imread(scanned_img_fp)  # Grid + noise/text, possibly rotated

# Preprocess: convert to grayscale and extract edges.
edgesA = preprocess(imgA)
edgesB = preprocess(imgB)

# Detect lines using Hough Transform.
linesA = detect_lines(edgesA)
linesB = detect_lines(edgesB)
print(f"Detected {len(linesA)} lines in document A and {len(linesB)} lines in document B.")
# %%
# let's visualize these lines
def draw_lines(image, lines, color=(0, 255, 0), thickness=2):
    """
    Draw lines on an image.
    """
    for x1, y1, x2, y2 in lines:
        cv2.line(image, (x1, y1), (x2, y2), color, thickness)
    
    return image

display(Image.fromarray(draw_lines(imgA.copy(), linesA)))
display(Image.fromarray(draw_lines(imgB.copy(), linesB)))

# %%
import cv2
import numpy as np
from sklearn.cluster import KMeans

def cluster_lines_kmeans(lines, n_clusters=2):
    """Cluster lines into two groups based on their angles using k-means."""
    angles = np.array([line_angle(line) for line in lines]).reshape(-1, 1)
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(angles)
    groups = [[] for _ in range(n_clusters)]
    for i, label in enumerate(kmeans.labels_):
        groups[label].append(lines[i])
    return groups

def compute_rotation_angle(group1, group2):
    """Compute rotation angle from two line groups."""
    angles = [line_angle(line) for line in group1 + group2]
    centroid1 = np.mean([line_angle(line) for line in group1])
    centroid2 = np.mean([line_angle(line) for line in group2])
    return (centroid1 + centroid2 - 90) / 2

def get_grid_points(lines_group1, lines_group2):
    """Compute intersections between two groups of lines."""
    intersections = []
    for line1 in lines_group1:
        for line2 in lines_group2:
            pt = compute_intersection(line1, line2)
            if pt is not None:
                intersections.append(pt)
    return np.array(intersections)

# Load images
imgA = cv2.imread(template_img_fp)
imgB = cv2.imread(scanned_img_fp)

# Preprocess and detect lines
edgesA = preprocess(imgA)
edgesB = preprocess(imgB)
linesA = detect_lines(edgesA)
linesB = detect_lines(edgesB)

# Cluster lines and compute rotation angles
groupsA = cluster_lines_kmeans(linesA)
groupsB = cluster_lines_kmeans(linesB)
rotationA = compute_rotation_angle(*groupsA)
rotationB = compute_rotation_angle(*groupsB)
delta_rotation = rotationB - rotationA

# Compute grid points and centroids
gridA = get_grid_points(*groupsA)
gridB = get_grid_points(*groupsB)
centroidA = np.mean(gridA, axis=0)
centroidB = np.mean(gridB, axis=0)

# Calculate scaling factors after rotation
rot_mat = cv2.getRotationMatrix2D(tuple(centroidB), -delta_rotation, 1)
rotated_gridB = cv2.transform(gridB[None], rot_mat)[0]
minA, maxA = np.min(gridA, axis=0), np.max(gridA, axis=0)
minB, maxB = np.min(rotated_gridB, axis=0), np.max(rotated_gridB, axis=0)
scale_x = (maxA[0] - minA[0]) / (maxB[0] - minB[0])
scale_y = (maxA[1] - minA[1]) / (maxB[1] - minB[1])

# Build affine transformation matrix
theta_rad = np.radians(-delta_rotation)
R = np.array([[np.cos(theta_rad), -np.sin(theta_rad)], 
              [np.sin(theta_rad), np.cos(theta_rad)]])
S = np.diag([scale_x, scale_y])
M_rot_scale = np.hstack([R.dot(S), [[0], [0]]])
M_translation = np.array([[1, 0, centroidA[0] - centroidB[0]*scale_x], 
                          [0, 1, centroidA[1] - centroidB[1]*scale_y]])
# Apply rotation and scaling first.
display(Image.fromarray(imgB))
temp = cv2.warpAffine(imgB, M_rot_scale, (imgA.shape[1], imgA.shape[0]))
# Then apply the translation.
display(Image.fromarray(temp))
aligned_imgB = cv2.warpAffine(temp, M_translation, (imgA.shape[1], imgA.shape[0]))
display(Image.fromarray(aligned_imgB))
# %%
# Display aligned image
display(Image.fromarray(cv2.cvtColor(aligned_imgB, cv2.COLOR_BGR2RGB)))
# %%
