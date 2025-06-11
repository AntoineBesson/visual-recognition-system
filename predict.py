import os
import joblib
import numpy as np
from skimage.io import imread
from skimage.filters import threshold_local
from skimage import measure
from skimage.measure import regionprops, find_contours, approximate_polygon
from skimage.transform import resize, warp, ProjectiveTransform
from skimage.morphology import opening, footprint_rectangle
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run_prediction(image_path="car.jpg"):
    """
    This function loads a trained model and an image, and predicts the license plate.
    """
    # 1. LOAD THE TRAINED MODEL
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'models/svc/svc.pkl')
    try:
        model = joblib.load(model_path)
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        return

    # 2. LOAD AND PRE-PROCESS THE IMAGE
    try:
        car_image = imread(image_path, as_gray=True)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    # --- Show the binarized image that the script "sees" ---
    # Restore to a more robust binarization: use a larger block_size and a higher offset
    block_size = 35
    local_thresh = threshold_local(car_image, block_size, offset=0.07)
    binary_car_image = car_image > local_thresh
    # Use a smaller, less aggressive morphological opening to preserve details
    binary_car_image = opening(binary_car_image, footprint_rectangle((3, 7)))
    label_image = measure.label(binary_car_image)

    fig_bin, ax_bin = plt.subplots(1, figsize=(12, 6))
    ax_bin.imshow(binary_car_image, cmap='gray')
    ax_bin.set_title("Binarized Image (What the script sees)")
    plt.show()

    # --- Show all candidate regions as red boxes on the original image ---
    fig_candidates, ax_candidates = plt.subplots(1, figsize=(12, 6))
    ax_candidates.imshow(car_image, cmap='gray')
    ax_candidates.set_title("All Plate Candidates (Red Boxes)")

    candidate_count = 0
    candidate_regions = []
    for region in regionprops(label_image):
        if region.area < 400:
            continue
        min_row, min_col, max_row, max_col = region.bbox
        rect_border = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row,
                                       edgecolor="red", linewidth=2, fill=False)
        ax_candidates.add_patch(rect_border)
        candidate_count += 1
        candidate_regions.append(region)
    if candidate_count == 0:
        print("No candidate regions found to display.")
    else:
        print(f"Displayed {candidate_count} candidate region(s) in red boxes.")

    plt.show()

    # Try relaxing the quadrilateral constraint and use bounding box warping if no plate is found
    output_shape = (50, 250)
    MIN_CHARACTERS = 3
    MAX_CHARACTERS = 10

    final_plate_image = None
    final_characters = []

    for region in candidate_regions:
        # Try to use quadrilateral if possible, else fallback to bbox
        plate_found = False
        try:
            contours = find_contours(region.convex_image, 0.5)
            if contours:
                poly = approximate_polygon(contours[0], tolerance=2)
                if len(poly) == 4:
                    abs_coords = poly + region.bbox[:2]
                    transform = ProjectiveTransform()
                    dst_coords = np.array([[0, 0], [0, output_shape[0]], [output_shape[1], output_shape[0]], [output_shape[1], 0]])
                    if transform.estimate(dst_coords, abs_coords):
                        # Use nearest-neighbor interpolation and preserve range to avoid gray pixels
                        warped_plate = warp(car_image, transform.inverse, output_shape=output_shape, order=0, preserve_range=True)
                        plate_found = True
            if not plate_found:
                # fallback: use bounding box
                min_row, min_col, max_row, max_col = region.bbox
                cropped = car_image[min_row:max_row, min_col:max_col]
                warped_plate = resize(cropped, output_shape, order=0, preserve_range=True, anti_aliasing=False)
            # Binarize after warping/resizing using a fixed threshold
            warped_plate = warped_plate.astype(float)
            binary_warped = warped_plate > 0.5
            license_plate_candidate = np.invert(binary_warped)

            labelled_candidate = measure.label(license_plate_candidate)
            character_dimensions = (
                0.35 * output_shape[0], 0.80 * output_shape[0],
                0.05 * output_shape[1], 0.20 * output_shape[1]
            )
            min_height, max_height, min_width, max_width = character_dimensions

            current_characters = []
            for char_region in regionprops(labelled_candidate):
                y0, x0, y1, x1 = char_region.bbox
                region_height = y1 - y0
                region_width = x1 - x0
                if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                    roi = license_plate_candidate[y0:y1, x0:x1].astype(float)
                    # Pad to square before resizing to 20x20 to avoid distortion
                    h, w = roi.shape
                    if h > w:
                        pad = ((0,0), ((h-w)//2, (h-w)-(h-w)//2))
                    else:
                        pad = (((w-h)//2, (w-h)-(w-h)//2), (0,0))
                    roi_padded = np.pad(roi, pad, mode='constant', constant_values=0)
                    resized_char = resize(roi_padded, (20, 20), order=0, preserve_range=True, anti_aliasing=False)
                    # Binarize the character image to ensure only 0 and 1 values
                    resized_char = (resized_char > 0.5).astype(np.uint8)
                    current_characters.append((x0, resized_char))

            if MIN_CHARACTERS <= len(current_characters) <= MAX_CHARACTERS:
                final_plate_image = license_plate_candidate
                final_characters = current_characters
                break

        except Exception:
            continue

    if final_plate_image is None:
        print("Could not find a valid license plate among the candidates.")
        return

    # 4. PREDICT CHARACTERS AND DISPLAY RESULT
    column_list = []
    for x0, char_img in final_characters:
        flat_char = char_img.reshape(-1)
        prediction = model.predict([flat_char])[0]
        column_list.append((x0, prediction))

    fig, ax = plt.subplots(1, figsize=(8, 3))
    ax.set_title("Final Plate & Predicted Characters")
    ax.imshow(final_plate_image, cmap='gray')

    # Draw rectangles around the segmented characters on the plate
    for x0, char_img in final_characters:
        # Find the region in the plate that matches this character's x0
        # (Assumes no overlap and sorted by x0)
        labelled_plate = measure.label(final_plate_image)
        character_dimensions = (
            0.35 * output_shape[0], 0.80 * output_shape[0],
            0.05 * output_shape[1], 0.20 * output_shape[1]
        )
        min_height, max_height, min_width, max_width = character_dimensions
        for region in regionprops(labelled_plate):
            y0, x0_r, y1, x1 = region.bbox
            region_height = y1 - y0
            region_width = x1 - x0_r
            if (
                region_height > min_height and region_height < max_height and
                region_width > min_width and region_width < max_width and
                abs(x0 - x0_r) < 3  # match by x0
            ):
                rect_border = patches.Rectangle((x0_r, y0), x1 - x0_r, y1 - y0, edgecolor="red", linewidth=2, fill=False)
                ax.add_patch(rect_border)
                break

    sorted_characters = sorted(column_list, key=lambda x: x[0])
    final_plate = ''.join([char for _, char in sorted_characters])

    print("\n---------------------------------")
    print(f"PREDICTED LICENSE PLATE: {final_plate}")
    print("---------------------------------")
    plt.tight_layout()
    plt.show()

# --- SCRIPT EXECUTION STARTS HERE ---
if __name__ == '__main__':
    run_prediction(image_path="car7.png")