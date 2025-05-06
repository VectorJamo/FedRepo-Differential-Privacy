
#This code convert the png data to pkl format, do not need it after converting image in pkl format
import os
import pickle
from PIL import Image
import numpy as np

def convert_folders_to_pickle(folder_paths, pickle_file_path):
    """
    Converts images from multiple folders into a pickle file with labels.

    Args:
        folder_paths (list): List of folder paths containing images.
        pickle_file_path (str): Path to save the output pickle file.
    """
    data = []
    labels = []

    # Process each folder
    for label, folder_path in enumerate(folder_paths):
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Process each image in the folder
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Open the image and convert it to a NumPy array
                    image = Image.open(file_path).convert('RGB')
                    image = image.resize((32,32))
                    image_data = np.array(image).transpose(2, 0, 1)  # Convert to (3, 32, 32)

                    # Append the image data and label
                    data.append(image_data)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    # Serialize the data and labels into a pickle file
    with open(pickle_file_path, 'wb') as pkl_file:
        pickle.dump({'data': data, 'labels': labels}, pkl_file)

    print(f"Images and labels saved to {pickle_file_path}")


def convert_folders_to_pickle(folder_paths, train_pickle_file, test_pickle_file, test_split=0.2):
    """
    Converts images from multiple folders into train and test pickle files with labels,
    and resizes images to (3, 32, 32).

    Args:
        folder_paths (list): List of folder paths containing images.
        train_pickle_file (str): Path to save the train pickle file.
        test_pickle_file (str): Path to save the test pickle file.
        test_split (float): Proportion of data to be used for testing (default is 0.2).
    """
    data = []
    labels = []

    # Target size for resizing
    target_size = (32, 32)

    # Process each folder
    for label, folder_path in enumerate(folder_paths):
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            continue

        # Process each image in the folder
        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                file_path = os.path.join(folder_path, file_name)
                try:
                    # Open the image, resize it, and ensure it has 3 channels
                    image = Image.open(file_path).convert('RGB')
                    image = image.resize(target_size)
                    image_data = np.array(image).transpose(2, 0, 1)  # Convert to (3, 32, 32)

                    # Append the image data and label
                    data.append(image_data)
                    labels.append(label)
                except Exception as e:
                    print(f"Error processing {file_name}: {e}")

    # Shuffle data and split into train and test sets
    indices = np.arange(len(data))
    np.random.shuffle(indices)

    split_idx = int(len(data) * (1 - test_split))
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_data = [data[i] for i in train_indices]
    train_labels = [labels[i] for i in train_indices]
    test_data = [data[i] for i in test_indices]
    test_labels = [labels[i] for i in test_indices]

    # Serialize the train data and labels into a pickle file
    with open(train_pickle_file, 'wb') as train_pkl_file:
        pickle.dump({'data': train_data, 'labels': train_labels}, train_pkl_file)

    # Serialize the test data and labels into a pickle file
    with open(test_pickle_file, 'wb') as test_pkl_file:
        pickle.dump({'data': test_data, 'labels': test_labels}, test_pkl_file)

    print(f"Train data and labels saved to {train_pickle_file}")
    print(f"Test data and labels saved to {test_pickle_file}")





data_dir = r"C:\Users\mdmor\OneDrive - Concordia University of Edmonton\CUE\Research\IEEEDataPort"
cur_dir = "./"

# Modify this to the directory where you have your data
# if not os.path.exists(data_dir):
data_dest = os.path.join(os.getcwd(), "raw-data") # *CODE MODIFIED HERE*

# Make sure to create sub-folders with these names and put the respective data there
tumor_fdir = os.path.join(data_dir, "augmentedMRI/augmented")
tumor_pkldir = os.path.join(data_dest, "tumordata")
# Specify the folder paths and output pickle file
folders = [
    os.path.join(tumor_fdir,"yes"),  # Replace with your actual folder paths
    os.path.join(tumor_fdir, "no"),
]

# pickle_file = os.path.join(tumor_pkldir,"tumor4train.pkl")  # Desired output pickle file name
# convert_folders_to_pickle(folders, pickle_file)

train_pickle_file = os.path.join(tumor_pkldir,"tumor2train.pkl")  # Desired train pickle file name
test_pickle_file = os.path.join(tumor_pkldir,"tumor2test.pkl")    # Desired test pickle file name

convert_folders_to_pickle(folders, train_pickle_file, test_pickle_file)

