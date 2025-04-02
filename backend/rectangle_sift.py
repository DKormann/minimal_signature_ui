# %%
import cv2
import numpy as np
from IPython.display import display
from PIL import Image
import matplotlib.pyplot as plt

# %%
shape = (64,64)
blank_image = np.zeros(shape, np.uint8)
display(Image.fromarray(blank_image))
# %%

def draw_rectangle(width, height, angle, dpi=100):
    """
    Draws a rectangle of given width and height, rotated by the specified angle (in degrees), 
    with a horizontal line at 1/3 from the bottom and a vertical line at 4/5 from the left 
    (both lines rotated by the same angle). Returns the image as a NumPy array.
    
    Parameters:
        width (float): The width of the rectangle (in coordinate units).
        height (float): The height of the rectangle (in coordinate units).
        dpi (int): The resolution of the figure in dots-per-inch (affects output array size).
        angle (float): The rotation angle in degrees.
        
    Returns:
        numpy.ndarray: The rendered image as an array of shape (H, W, 3).
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    # Convert angle to radians for rotation calculations.
    rad = np.deg2rad(angle)
    
    # Helper function to rotate a point (x, y) around the origin (0,0)
    def rotate_point(x, y):
        return (x * np.cos(rad) - y * np.sin(rad), x * np.sin(rad) + y * np.cos(rad))
    
    # Draw the rectangle with rotation (rotated about the lower-left corner)
    rect = plt.Rectangle((0, 0), width, height, fill=False, edgecolor='black', 
                         linewidth=2, angle=angle)
    ax.add_patch(rect)
    
    # Calculate positions for the lines in the unrotated coordinate system
    y_line = height / 3         # horizontal line: 1/3 up from the bottom
    x_line = (4/5) * width      # vertical line: 4/5 from the left
    
    # Rotate endpoints for the horizontal line (from (0, y_line) to (width, y_line))
    h_start = rotate_point(0, y_line)
    h_end   = rotate_point(width, y_line)
    
    # Rotate endpoints for the vertical line (from (x_line, 0) to (x_line, height))
    v_start = rotate_point(x_line, 0)
    v_end   = rotate_point(x_line, height)
    
    # Draw the rotated lines
    ax.plot([h_start[0], h_end[0]], [h_start[1], h_end[1]], color='black', linewidth=2)
    ax.plot([v_start[0], v_end[0]], [v_start[1], v_end[1]], color='black', linewidth=2)
    
    # Determine the bounding box by rotating all four corners of the rectangle
    corners = [rotate_point(0, 0), rotate_point(width, 0),
               rotate_point(width, height), rotate_point(0, height)]
    xs, ys = zip(*corners)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    
    # Add a margin (10% of the bounding box width/height) to the limits
    margin_x = 0.1 * (max_x - min_x)
    margin_y = 0.1 * (max_y - min_y)
    ax.set_xlim(min_x - margin_x, max_x + margin_x)
    ax.set_ylim(min_y - margin_y, max_y + margin_y)
    
    ax.set_aspect('equal', adjustable='box')
    ax.axis('off')  # Remove axes for a clean image
    
    # Render the figure and convert to a NumPy array
    fig.canvas.draw()
    image_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    image_array = image_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)  # Free memory
    return image_array


rec_img_config = {
    "shape": (128,128),
    "width": 400,
    "height": 500,
    "color": (255,0,0),
    "thickness": 2,
    "angle": 0,
}

ratio = 8/10

# the image is same size, but the rec_dim are 10% smaller
smaller_rot_rec_img_config = rec_img_config.copy()
smaller_rot_rec_img_config["width"] = smaller_rot_rec_img_config["width"] * ratio
smaller_rot_rec_img_config["height"] = smaller_rot_rec_img_config["height"] * ratio
smaller_rot_rec_img_config["angle"] = 30

rectangle_image_1 = draw_rectangle(width=rec_img_config["width"], height=rec_img_config["height"], angle=rec_img_config["angle"])
display(Image.fromarray(rectangle_image_1))

# notice that the square is slightly smaller
rectangle_image_2 = draw_rectangle(width=smaller_rot_rec_img_config["width"], height=smaller_rot_rec_img_config["height"], angle=smaller_rot_rec_img_config["angle"])
display(Image.fromarray(rectangle_image_2))
# %%
# Initialize SIFT detector
sift = cv2.SIFT_create()

def filter_matches(matches):
    good_matches = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:
            good_matches.append(m)

    return good_matches

def find_matches(img_1, img_2):
    # Detect keypoints and compute descriptors
    kp1, des1 = sift.detectAndCompute(img_1, None)
    kp2, des2 = sift.detectAndCompute(img_2, None)

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
    result_img = cv2.drawMatches(rectangle_image_1, kp1, rectangle_image_2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    display(Image.fromarray(result_img))

    return good_matches, kp1, kp2

good_matches, kp1, kp2 = find_matches(rectangle_image_1, rectangle_image_2)

# %%
# let's visualize the keypoints
def visualize_keypoints(img, kp):
    img = cv2.drawKeypoints(img, kp, None)
    display(Image.fromarray(img))

visualize_keypoints(rectangle_image_1, kp1)
visualize_keypoints(rectangle_image_2, kp2)
# %%
# let's compute the transformation matrix
def trans_matrix(matches, kp1, kp2):
    # We need at least 4 matches to compute homography
    if len(matches) < 4:
        print("Not enough matches to compute homography.")
        return None, None

    # Extract the coordinates of the matched keypoints.
    src_pts = np.float32([ kp1[m.queryIdx].pt for m in matches ]).reshape(-1, 1, 2)
    dst_pts = np.float32([ kp2[m.trainIdx].pt for m in matches ]).reshape(-1, 1, 2)

    # Compute the inliers using RANSAC.
    M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return M, inliers

M, inliers = trans_matrix(good_matches, kp1, kp2)
# %%
import matplotlib.pyplot as plt

def create_image_grid(images, show_axis=True):
    amt_images = len(images)
    if amt_images > 9:
        raise ValueError("Can only visualize up to 9 images at once.")
    
    # we want a max of 3 columns.
    amt_cols = min(3, amt_images)
    amt_rows = int(np.ceil(amt_images / amt_cols))

    fig = plt.figure()
    
    for i, image in enumerate(images):
        # Iterating over the grid returns the Axes.
        ax = fig.add_subplot(amt_rows, amt_cols, i + 1)
        ax.imshow(image)

        if not show_axis:
            ax.axis('off')

    # adjust spacing between subplots.
    plt.tight_layout()
    
    plt.show()

def project_image(scan_img, templ_img, M):
    projected_image = cv2.warpAffine(scan_img, M, (templ_img.shape[1], templ_img.shape[0]))

    return projected_image

projected_image = project_image(scan_img=rectangle_image_1, templ_img=rectangle_image_2, M=M)
images = [np.array(rectangle_image_1), np.array(rectangle_image_2), np.array(projected_image)]

titles = ["big rectangle", "small rectangle", "warped rectangle"]

create_image_grid(images, titles)
# %%
