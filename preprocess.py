import glob
import os
import cv2
import numpy as np

dataset_root = 'datasets/cityscapes'
dataset_label = 'datasets/cityscapes/gtFine_trainvaltest/gtFine'
dataset_image = 'datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit'

# Get filenames
train_labels = glob.glob(os.path.join(dataset_label, 'train/*/*labelIds.png'), recursive=True)
val_labels = glob.glob(os.path.join(dataset_label, 'val/*/*labelIds.png'), recursive=True)
train_labels = np.sort(train_labels)
val_labels = np.sort(val_labels)

train_images = glob.glob(os.path.join(dataset_image, 'train/*/*.png'), recursive=True)
val_images = glob.glob(os.path.join(dataset_image, 'val/*/*.png'), recursive=True)
train_images = np.sort(train_images)
val_images = np.sort(val_images)

# Generate training dataset
road = 7
os.makedirs(os.path.join(dataset_root, 'train'), exist_ok=True)
for i in range(len(train_images)):
    print(i, '/', len(train_images))

    # Read data
    img = cv2.imread(train_images[i])
    label = cv2.imread(train_labels[i])

    # Extract road region
    label = np.where(label == road, 255, 0)

    # Combine input image and ground truth
    h, w = img.shape[:-1]
    comb = np.zeros((h, w * 2, 3), np.uint8)
    comb[:, :w, :] = img
    comb[:, w:, :] = label

    # Save image
    cv2.imwrite(os.path.join(dataset_root, 'train/' + '{0:04d}'.format(i) + '.png'), comb)

# Generate validation dataset
os.makedirs(os.path.join(dataset_root, 'val'), exist_ok=True)
for i in range(len(val_images)):
    print(i, '/', len(val_images))

    # Read data
    img = cv2.imread(val_images[i])
    label = cv2.imread(val_labels[i])

    # Extract road region
    label = np.where(label == road, 255, 0)

    # Combine input image and ground truth
    h, w = img.shape[:-1]
    comb = np.zeros((h, w * 2, 3), np.uint8)
    comb[:, :w, :] = img
    comb[:, w:, :] = label

    # Save image
    cv2.imwrite(os.path.join(dataset_root, 'val/' + '{0:04d}'.format(i) + '.png'), comb)