import os
import joblib
import numpy as np
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def run_prediction(image_path="car.jpg"):
    """
    This function loads a trained model and an image, and predicts the license plate.
    """
    # ==============================================================================
    # 1. LOAD THE TRAINED MODEL
    # ==============================================================================
    print("Loading trained model...")
    current_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(current_dir, 'models/svc/svc.pkl')
    
    try:
        model = joblib.load(model_path)
        print("Model loaded successfully.")
    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}")
        print("Please ensure you have run recognition.py to train and save the model.")
        return

    # ==============================================================================
    # 2. LOAD AND PRE-PROCESS THE IMAGE
    # ==============================================================================
    print(f"Processing image: {image_path}...")
    try:
        car_image = imread(image_path, as_gray=True)
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return

    gray_car_image = car_image * 255
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    ax1.set_title("Original Image")
    ax1.imshow(gray_car_image, cmap="gray")

    threshold_value = threshold_otsu(gray_car_image)
    binary_car_image = gray_car_image > threshold_value
    ax2.set_title("Detected Plate Candidates")
    ax2.imshow(binary_car_image, cmap="gray")

    # ==============================================================================
    # 3. DETECT LICENSE PLATE CANDIDATES
    # ==============================================================================
    label_image = measure.label(binary_car_image)
    plate_dimensions = (0.08 * label_image.shape[0], 0.2 * label_image.shape[0], 0.15 * label_image.shape[1], 0.4 * label_image.shape[1])
    min_height, max_height, min_width, max_width = plate_dimensions

    plate_like_objects = []
    plate_like_bboxes = []
    for region in regionprops(label_image):
        if region.area < 50:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        region_height = max_row - min_row
        region_width = max_col - min_col

        # Heuristic: aspect ratio and mean intensity
        aspect_ratio = region_width / region_height if region_height > 0 else 0
        candidate_img = binary_car_image[min_row:max_row, min_col:max_col]
        mean_intensity = np.mean(gray_car_image[min_row:max_row, min_col:max_col])

        # Typical license plate: aspect ratio between 2 and 6, not too bright
        if (
            region_height >= min_height and region_height <= max_height and
            region_width >= min_width and region_width <= max_width and
            region_width > region_height
        ):
            candidate_img = binary_car_image[min_row:max_row, min_col:max_col]
            # Compute vertical projection profile (sum of dark pixels per column)
            # Invert so that dark pixels (characters) are 1, background is 0
            candidate_inverted = np.invert(candidate_img.astype(bool)).astype(int)
            vertical_profile = np.sum(candidate_inverted, axis=0)
            profile_std = np.std(vertical_profile)
            # Heuristic: plates with characters have higher std in vertical profile
            if profile_std > 5:
                plate_like_objects.append(candidate_img)
                plate_like_bboxes.append((min_row, min_col, max_row, max_col))
                rect_border = patches.Rectangle((min_col, min_row), max_col - min_col, max_row - min_row, edgecolor="red", linewidth=2, fill=False)
                ax2.add_patch(rect_border)

    if not plate_like_objects:
        print("No objects matching license plate dimensions found.")
        plt.show()
        return
        
    # ==============================================================================
    # 4. VALIDATE AND SELECT THE CORRECT LICENSE PLATE USING SEGMENTATION
    # ==============================================================================
    print(f"\nFound {len(plate_like_objects)} potential license plate(s). Validating...")
    MIN_CHARACTERS = 3
    MAX_CHARACTERS = 10

    selected_plate_image = None
    selected_characters = None
    selected_columns = None

    for i, plate in enumerate(plate_like_objects):
        inverted_plate = np.invert(plate)
        labelled_plate = measure.label(inverted_plate)
        character_dimensions = (
            0.35 * inverted_plate.shape[0],
            0.60 * inverted_plate.shape[0],
            0.05 * inverted_plate.shape[1],
            0.15 * inverted_plate.shape[1]
        )
        min_height, max_height, min_width, max_width = character_dimensions

        characters = []
        columns = []
        for regions in regionprops(labelled_plate):
            y0, x0, y1, x1 = regions.bbox
            region_height = y1 - y0
            region_width = x1 - x0

            if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
                roi = inverted_plate[y0:y1, x0:x1]
                resized_char = resize(roi, (20, 20))
                characters.append((x0, resized_char))
                columns.append(x0)

        print(f"-> Candidate {i}: Found {len(characters)} segmented character(s).")
        if MIN_CHARACTERS <= len(characters) <= MAX_CHARACTERS:
            print(f"   Candidate {i} accepted as the license plate.")
            selected_plate_image = inverted_plate
            selected_characters = characters
            selected_columns = columns
            break
        else:
            print(f"   Candidate {i} rejected.")

    if selected_plate_image is None or selected_characters is None:
        print("\nCould not find a valid license plate among the candidates.")
        plt.show()
        return

    # ==============================================================================
    # 5. SEGMENT AND PREDICT CHARACTERS
    # ==============================================================================
    fig, ax3 = plt.subplots(1, figsize=(8, 3))
    ax3.set_title("Segmented Characters on Selected Plate")
    ax3.imshow(selected_plate_image, cmap='gray')

    column_list = []
    for x0, char_img in selected_characters:
        flat_char = char_img.reshape(-1)
        prediction = model.predict([flat_char])[0]
        column_list.append((x0, prediction))
        # Draw rectangle for visualization
        # Find the bounding box for this character
        # Since we have only x0, estimate y0, y1, x1 from the mask
        # Instead, let's re-calculate the bbox from the regionprops for accuracy
    # Draw rectangles for all detected characters
    labelled_plate = measure.label(selected_plate_image)
    character_dimensions = (
        0.35 * selected_plate_image.shape[0],
        0.60 * selected_plate_image.shape[0],
        0.05 * selected_plate_image.shape[1],
        0.15 * selected_plate_image.shape[1]
    )
    min_height, max_height, min_width, max_width = character_dimensions
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0
        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red", linewidth=2, fill=False)
            ax3.add_patch(rect_border)

    # ==============================================================================
    # 6. DISPLAY THE FINAL RESULT
    # ==============================================================================
    sorted_characters = sorted(column_list, key=lambda x: x[0])
    final_plate = ''.join([char for _, char in sorted_characters])

    print("\n---------------------------------")
    print(f"PREDICTED LICENSE PLATE: {final_plate}")
    print("---------------------------------")

    plt.tight_layout()
    plt.show()

# --- SCRIPT EXECUTION STARTS HERE ---
if __name__ == '__main__':
    # You can change the image path here
    run_prediction(image_path="car.jpg")