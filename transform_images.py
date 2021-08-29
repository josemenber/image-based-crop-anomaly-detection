from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os
import cv2
import pytz
import glob
import random
import collections


# Script arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Create RGB or NRG images for Agriculture-Vision 2020 dataset.')
    parser.add_argument("--nrg", action='store_true', default=False, help="Create NRG images. By default create RGB images.")
    args = parser.parse_args()
    return args.nrg

# Read arguments
(nrg) = parse_args()

# Global params
path = './Agriculture-Vision/'
pathTrain = './train_images/'
pathVal = './val_images/'


classes = ['cloud_shadow', 'double_plant', 'planter_skip', 'standing_water', 'waterway', 'weed_cluster', 'background']
num_classes = [0, 1, 2, 3, 4, 5, 6]

# Delete train and validation image directories
for root, dirs, files in os.walk(pathTrain, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
        
for root, dirs, files in os.walk(pathVal, topdown=False):
    for name in files:
        os.remove(os.path.join(root, name))
    for name in dirs:
        os.rmdir(os.path.join(root, name))
        
        
# Create train and validation image directories
if not os.path.exists(pathTrain):
    os.makedirs(pathTrain)

if not os.path.exists(pathVal):
    os.makedirs(pathVal)
    

# Create train and validation images
for subset in ['train', 'val']:
    filename = []
    specie = []
    label = []
    for file in glob.glob(path + subset + '/images/rgb/*.jpg'):
        # Extract the name of the image
        basename = os.path.basename(file).split('.')[0]
        
        image = cv2.imread(file)
        
        if(nrg):
            nir_image = cv2.imread(path + subset + '/images/nir/' + basename + '.jpg')
            # Extract N, R y G channels
            (B, G, R) = cv2.split(image)
            (N, I, R1) = cv2.split(nir_image)
            image = cv2.merge([N, R, G])
            
        # Read boundarie and mask
        boundarie_image = cv2.imread(path + subset + '/boundaries/' + basename + '.png', 0)
        mask_image = cv2.imread(path + subset + '/masks/' + basename + '.png', 0)
        
        # Created anomaly images
        for i in range(0, 6):
            label_image = cv2.imread(path + subset + '/labels/' + classes[i] + '/' + basename + '.png', 0)
            
            # if not all the image is black
            if cv2.countNonZero(label_image) > 0:
                result_boundarie = cv2.bitwise_and(image, image, mask = boundarie_image)
                result_bound_and_mask = cv2.bitwise_and(result_boundarie, result_boundarie, mask = mask_image)
                result_bound_mask_and_label = cv2.bitwise_and(result_bound_and_mask, result_bound_and_mask, mask = label_image)
                        
                # New name for the image
                new_filename = basename + '-' + classes[i] + '.jpg'
                # save images in the train/validation folder
                if(subset == 'train'):
                    cv2.imwrite(pathTrain + new_filename, result_bound_mask_and_label)
                else:
                    cv2.imwrite(pathVal + new_filename, result_bound_mask_and_label)

                filename.append(new_filename)
                specie.append(classes[i])
                label.append(num_classes[i])
        
        # Created background images
        mask_image = cv2.bitwise_not(cv2.imread(path + subset + '/labels/cloud_shadow/' + basename + '.png', 0))
        mask1_image = cv2.bitwise_not(cv2.imread(path + subset + '/labels/double_plant/' + basename + '.png', 0))
        mask2_image = cv2.bitwise_not(cv2.imread(path + subset + '/labels/planter_skip/' + basename + '.png', 0))
        mask3_image = cv2.bitwise_not(cv2.imread(path + subset + '/labels/standing_water/' + basename + '.png', 0))
        mask4_image = cv2.bitwise_not(cv2.imread(path + subset + '/labels/waterway/' + basename + '.png', 0))
        mask5_image = cv2.bitwise_not(cv2.imread(path + subset + '/labels/weed_cluster/' + basename + '.png', 0))

        result = cv2.bitwise_and(mask_image, mask_image, mask = mask1_image)
        result1 = cv2.bitwise_and(result, result, mask = mask2_image)
        result2 = cv2.bitwise_and(result1, result1, mask = mask3_image)
        result3 = cv2.bitwise_and(result2, result2, mask = mask4_image)
        result4 = cv2.bitwise_and(result3, result3, mask = mask5_image)
        
        boundarie_image = cv2.imread(path + subset + '/boundaries/' + basename + '.png', 0)
        mask_image = cv2.imread(path + subset + '/masks/' + basename + '.png', 0)

        result_boundarie = cv2.bitwise_and(result4, result4, mask = boundarie_image)
        result_bound_and_mask = cv2.bitwise_and(result_boundarie, result_boundarie, mask = mask_image)

        # if not all the image is black
        if cv2.countNonZero(result_bound_and_mask) > 0:
            image = cv2.imread(file)
            
            if(nrg):
                nir_image = cv2.imread(path + subset + '/images/nir/' + basename + '.jpg')
                # Extract N, R y G channels
                (B, G, R) = cv2.split(image)
                (N, I, R1) = cv2.split(nir_image)
                image = cv2.merge([N, R, G])
        
            result_background = cv2.bitwise_and(image, image, mask = result_bound_and_mask)
            new_filename = basename + '-' + classes[6] + '.jpg'
            # save images in the train/validation folder
            if(subset == 'train'):
                cv2.imwrite(pathTrain + new_filename, result_background)
            else:
                cv2.imwrite(pathVal + new_filename, result_background)
            
            filename.append(new_filename)
            specie.append(classes[6])
            label.append(num_classes[6])

    # save information as csv
    d = {'Filename':filename, 'Label':label, 'Species':specie}
    df = pd.DataFrame(d)
    df.to_csv(subset + '.csv', index=False)


