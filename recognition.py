import os
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
import joblib # Using the modern import for joblib
from skimage.io import imread
from skimage.filters import threshold_otsu

# Define the characters we are training for
letters = [
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D',
    'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T',
    'U', 'V', 'W', 'X', 'Y', 'Z'
]

def read_training_data(training_directory):
    image_data = []
    target_data = []
    for each_letter in letters:
        # The tutorial uses 10 images per character, you may have more or less.
        # This part looks for files like 'A_0.jpg', 'A_1.jpg', etc.
        # You may need to adjust this loop depending on your dataset.
        for each in range(10): # Assumes 10 images per character
            image_path = os.path.join(training_directory, each_letter, f"{each_letter}_{each}.jpg")
            # read each image of each character
            img_details = imread(image_path, as_gray=True)
            # converts each character image to binary image
            binary_image = img_details < threshold_otsu(img_details)
            # flatten the 2D image array to a 1D array
            flat_bin_image = binary_image.reshape(-1)
            image_data.append(flat_bin_image)
            target_data.append(each_letter)

    return (np.array(image_data), np.array(target_data))

def cross_validation(model, num_of_fold, train_data, train_label):
    # This uses the concept of cross-validation to measure the accuracy of a model 
    accuracy_result = cross_val_score(model, train_data, train_label,
                                      cv=num_of_fold)
    print(f"Cross Validation Result for {num_of_fold}-fold")
    print(accuracy_result * 100)

# --- Main script execution starts here ---

print("Reading training data...")
current_dir = os.path.dirname(os.path.realpath(__file__))
training_dataset_dir = os.path.join(current_dir, 'train')

image_data, target_data = read_training_data(training_dataset_dir)
print("Training data read successfully.")

# Create the SVC model as specified in the tutorial 
svc_model = SVC(kernel='linear', probability=True)

print("Performing cross-validation...")
# Perform 4-fold cross-validation 
cross_validation(svc_model, 4, image_data, target_data)

print("Training model on the entire dataset...")
# Train the model with all the input data 
svc_model.fit(image_data, target_data)

print("Model trained successfully. Saving model...")
# Persist the trained model to a file 
save_directory = os.path.join(current_dir, 'models/svc/')
if not os.path.exists(save_directory):
    os.makedirs(save_directory)

joblib.dump(svc_model, os.path.join(save_directory, 'svc.pkl'))
print(f"Model saved to: {os.path.join(save_directory, 'svc.pkl')}")