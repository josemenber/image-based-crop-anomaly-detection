from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Model, load_model
from keras import backend as K
import pandas as pd
import numpy as np
import argparse
import csv
import os
import cv2
import glob


SEED_VALUE = 1  # Set the seed for reproducibility


# Script arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate DeepWeeds or Agriculture-Vision models.')
    parser.add_argument("--agriculture", action='store_true', default=False, help="Evaluate models in Agriculture-Vision 2020 dataset. By default use DeepWeeds dataset.")
    args = parser.parse_args()
    return args.agriculture

# Read arguments
(agriculture) = parse_args()


if(agriculture):
    test_dataframe = pd.read_csv("val.csv")
    img_size = (512, 512)
    folder = './val_images/'
    
else:
    df = pd.read_csv("./labels/labels.csv")

    # Split data: train (60%), validation(20%) and test (20%)
    train_index, test_index = train_test_split(df.index, test_size=0.4, random_state=SEED_VALUE, shuffle=True, stratify=df['Label'])
    val_index, test_index = train_test_split(test_index, test_size=0.5, random_state=SEED_VALUE, shuffle=True, stratify=df.loc[test_index, 'Label'])

    test_dataframe = df.loc[test_index, ]
    img_size = (224, 224)
    folder = './images/'
    

# No testing image augmentation (except for converting pixel values to floats)
test_data_generator = ImageDataGenerator(rescale=1./255)

# Load test images in batches from directory and apply rescaling
test_data_generator = test_data_generator.flow_from_dataframe(test_dataframe,
                                                              directory=folder,
                                                              x_col="Filename",
                                                              y_col="Species",
                                                              target_size=img_size,
                                                              batch_size=32,
                                                              shuffle=False,
                                                              seed=SEED_VALUE,
                                                              class_mode='categorical')


print("Class index:")
print(test_data_generator.class_indices)

# Evaluating all models in the models folder
for file in glob.glob("./models/*"):
    print(str(file))
    print()
    # Load the best model
    model = load_model(file)

	# Evaluate model on test subset
    predictions = model.predict_generator(test_data_generator, len(test_dataframe.index)//32+1)
    y_true = test_data_generator.classes
    y_pred = np.argmax(predictions, axis=1)

	# Generate and print classification metrics and confusion matrix
    print()
    print(accuracy_score(y_true, y_pred))
    print()
    print(confusion_matrix(y_true, y_pred))
    print()
    print(classification_report(y_true, y_pred))

