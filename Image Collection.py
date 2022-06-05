
import os
import sys
import time
sys.path.insert(0, "./libraries")
from libraries.TFOD import TFOD, TFPP

# labels = ["peace", "thumbsdown", "middlefinger"]
# This is the labels you want to train your model for.
labels = ["wirestripper"]
number_imgs = 15

# Path to where your images taken on the webcam will be stored.
# IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images", "collectedimages")

# Create the class to deal with model setup.
# tfpp = TFPP(labels, number_imgs, IMAGES_PATH)
#
# # Ensures creation of folders to house webcam images of labels
# tfpp.setup_folders()
#
# # Operator to hit SPACE key to capture images of labels. This will get stored in correct folder
# tfpp.capture_images()
#
# # # Opens a UI to label images stored in collectedimages folder. Open the folder you want and label all images.
# tfpp.label_images()
#
IMAGES_PATH = os.path.join("Tensorflow", "workspace", "images")
tfod = TFOD(labels, IMAGES_PATH)
# tfod.check_paths()
# tfod.create_label_maps()
# tfod.create_tf_records()
# Move pipeline config from workspace/pre-training to workspace/models then run this
# tfod.update_config_for_learning()
# tfod.train_tf_model()
# tfod.evaluate_model()
# tfod.detect_image()
tfod.detect_real_time_obj()

