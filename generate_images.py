import pandas as pd
import os
import random
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img


SEED_VALUE = 1  # Set the seed for reproducibility


# Script arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Generating new anomaly images with or without gaussian noise.')
    parser.add_argument("--agriculture", action='store_true', default=False, help="Generate images for the Agriculture-Vision 2020 dataset. By default it uses the DeepWeeds dataset.")
    parser.add_argument("--noise", action='store_true', default=False, help="Apply noise to images.")
    parser.add_argument('--num_imgs', type=int, default=1, help="Number of images to be generated for each initial anomaly image.")
    args = parser.parse_args()
    return args.agriculture, args.noise, args.num_imgs

# Read arguments
(agriculture, gaussian_noise, num_imgs) = parse_args()


# Agriculture dataset
if(agriculture):
    num_classes = [0, 1, 2, 3, 4, 5, 6]
    classes = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster', 'background']
    
    # Read training file
    df = pd.read_csv("train.csv")
    test_size=0.2
    
    # Params
    img_directory = './train_images/'
    zoom = 0.0 # No zoom
    
else:
    num_classes = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    classes = ['Chinee apple', 'Lantana', 'Parkinsonia', 'Parthenium', 'Prickly acacia', 'Rubber vine', 'Siam weed', 'Snake weed', 'Negative']
    
    # Read label file
    df = pd.read_csv("labels/labels.csv")
    test_size=0.4

    # Params
    img_directory = './images/'
    zoom = (0.8, 1.2)
    

# Select the train images
train_index, val_index = train_test_split(df.index, test_size=test_size, random_state=SEED_VALUE, shuffle=True, stratify=df['Label'])
train_dataframe = df.loc[train_index, ]

# Create new image directories
directories = ['./new_images/'] + ['./new_images/' + str(i) + '/' for i in range(0, max(num_classes))]

# Add gaussian noise function
# param img: Input image
# return: img with gaussian noise
def add_noise(image):
    row,col,ch = image.shape
    mean = 0
    var = 0.1
    sigma = var**0.5
    gauss = np.random.normal(mean,sigma, (row,col,ch))
    gauss = gauss.reshape(row, col, ch)
    noisy = image + gauss
    return noisy

# Delete (if exits) and create new image directories
for root, dirs, files in os.walk('./new_images/', topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))

for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)




# Image augmentation with and without gaussian noise
if(gaussian_noise):
    data_generator = ImageDataGenerator(fill_mode="constant",
                                        shear_range=0.2,
                                        zoom_range=zoom,
                                        horizontal_flip=True,
                                        rotation_range=360,
                                        channel_shift_range=25,
                                        brightness_range=(0.8, 1.2),
                                        preprocessing_function=add_noise)
else:
    data_generator = ImageDataGenerator(fill_mode="constant",
                                        shear_range=0.2,
                                        zoom_range=zoom,
                                        horizontal_flip=True,
                                        rotation_range=360,
                                        channel_shift_range=25,
                                        brightness_range=(0.8, 1.2))

# Generating new anomaly images
if(num_imgs > 0):
    for i in range(0, max(num_classes)):
        df1 = train_dataframe[train_dataframe['Label'] == i]
        for j in range(0, len(df1.index)):
            img = load_img(img_directory + df1['Filename'].iloc[j])
            x = img_to_array(img)
            x = np.expand_dims(x, axis=0)

            n = 0
            for batch in data_generator.flow(x, batch_size=1, save_to_dir='./new_images/'+str(i)+'/', save_format='jpg'):
                n += 1
                if(n >= num_imgs):
                    break
                
# Creating new dataframe with new images information
new_df = pd.DataFrame(columns=['Filename', 'Label', 'Species'])
for clase in range(0, max(num_classes)):
    names = os.listdir('./new_images/'+str(clase)+'/')
    names = ["{}{}".format('../new_images/'+str(clase)+'/',i) for i in names]
    labels = [classes[clase]]*len(names)
    numlabels = [clase]*len(names)
    d = {'Filename':names,'Label':numlabels, 'Species':labels}
    df_aux = pd.DataFrame(d)
    new_df = pd.concat([new_df, df_aux])
    
# Save new images information as csv
new_df.to_csv('new_dataframe.csv', index=False)

