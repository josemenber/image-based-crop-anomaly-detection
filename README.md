# Image-based crop anomaly detection using Artificial Intelligence techniques

This repository makes available the source code and methodology of the work: "A general methodology for image-based anomaly detection in smart agriculture". This paper presents a methodology for the classification of crop anomalies at the image level. Evaluated the performance of foue deep learning models such as VGG16, Inception-v3, ResNet-50 and Xception on the DeepWeeds and Agriculture-Vision challenge 2020 datasets. These models achieved an average classification accuracy of 95.6% and 96.6% respectively on the DeepWeeds dataset, improving on the 95.1% and 95.7% results obtained by [1]. For the Agriculture-Vision 2020 challenge dataset [2], an average classification accuracy of 95.3% was achieved with Inception-v3 and 94.3% with ResNet-50. In addition, the inference times of the models were calculated with an average time of 10 ms for DeepWeeds and 50 ms for the Agriculture-Vision dataset. This demonstrates that the real-time implementation of such complex models for the automatic detection of crop anomalies is feasible.

The contents of this repository are released under an [Apache 2 license](https://github.com/josemenber/image-based-crop-anomaly-detection/blob/master/LICENSE).

## Data organization

Images of DeepWeeds dataset are assigned unique filenames that include the date/time the image was photographed and an ID number for the instrument which produced the image. The format is like so: `YYYYMMDD-HHMMSS-ID`, where the ID is simply an integer from 0 to 3. The unique filenames are strings of 17 characters, such as 20170320-093423-1.

The `labels.csv` file from the labels folder assigns species labels to each image. It is a comma separated text file in the format:

```
Filename,Label,Species
...
20170207-154924-0,jpg,7,Snake weed
20170610-123859-1.jpg,1,Lantana
20180119-105722-1.jpg,8,Negative
...
```

The images of the Agriculture-Vision 2020 challenge dataset are divided into 3 folders: training, validation and test. Each folder has binary masks to select the valid pixels of each image, and also the regions containing the anomalies. The test folder does not have the labels because the predictions are used to evaluate the performance of the models.

## Download images

Images from the DeepWeeds dataset are available in the `images` folder. These images can also be downloaded by running the `download_images.py` script. The script does not receive any parameters and is executed:

```
    python3 download_images.py
```

This code downloads a zipped file with the images from the dataset and unzips it into an `images` folder like the one in the repository.

The Agriculture-Vision 2020 challenge dataset [2] is private and images cannot be included. However, they can be obtained by contacting the contest partners as explained [here](https://www.agriculture-vision.com/agriculture-vision-2020/dataset). Once obtained, they should be included in a folder called "Agriculture-Vision" with subfolders for each subset of data, as provided. Since the contest images are designed to perform pixel-level classification of anomalies, they need to be transformed to perform image-level classification. To do this, the `transform_images.py` script is used. This script is given the `--nrg` option to create the images with the NRG channels. By default uses RGB images. For help on how to run this script run the command using the `-h` option.

## Download and evaluate models

We provide the most successful ResNet50 and InceptionV3 models saved in Keras hdf5 model format for each dataset. Due to the size of the models they are hosted outside of the Github repository. The models must be downloaded:

* [DeepWeeds models](https://drive.google.com/file/d/1tGmL8fX2pdlkjepKWdfw_I1pFpRXnd2S/view?usp=sharing) (496.7MB).

* [Agriculture-Vision models](https://drive.google.com/file/d/1SIasRRfTW9E10Z0jUbyx1ygNv14xpnaj/view?usp=sharing) (504.7 MB). 

When you download the file you have to unzip it inside the repository folder. Once unzipped you can test the models by running the command:

```
    python3 eval_models.py
```

to evaluate the models on the DeepWeeds dataset, or by adding the `--agriculture` parameter to evaluate the performance of the models on the Agriculture-Vision dataset. For help on how to run this script run the `python3 eval_models.py -h` command.


## Generate new anomaly images

To generate new anomalies images by applying transformations to the initial images (expansion of the training set) the Python script `generate_images.py` has been created. This script receives 3 parameters:

* `--agriculture`: to generate images for the Agriculture-Vision 2020 dataset. By default it uses the DeepWeeds dataset

* `--noise`: to add gaussian noise to the images (default is not applied).
    
* `--num_imgs N`: where N is an integer indicating the number of images for each initial anomaly image. By default this value is 1.


For help on how to run this script execute the command:

```
    python3 generate_images.py -h
```

## Train and test models

To train the models and evaluate their performance on both datasets, the Python script `train_models.py` is used. This script receives 5 parameters:

* `--agriculture`: to train the models on the Agriculture-Vision dataset. By default the DeepWeeds dataset is used.

* `--noise`: to add Gaussian noise to the images (by default not applied).

* `--augmentation`: to add the new images previously generated with the *generate_imapges.py* script (not applied by default).

* `--model modelname`: to indicate the model name. *Modelname* takes the values: 'vgg16', 'resnet-50', 'inception-v3' or 'xception' (default).
    
* `--max_epochs N`: where N is an integer indicating the maximum number of epochs for training.

For help on how to run this script, run the command:

```
    python3 train_models.py -h
```


## Dependencies

The required Python packages to execute the scripts are listed in `requirements.txt` file.

## Citation

[1] A. Olsen, D. A. Konovalov, B. Philippa, P. Ridd, J. C. Wood, J. Johns, W. Banks, B. Girgenti, O. Kenny, J. Whinney, B. Calvert, M. Rahimi Azghadi, and R. D. White, DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning, *Scientific Reports*, vol. 9, no. 2058, **2** 2019. Available: https://doi.org/10.1038/s41598-018-38343-3

[2] Mang Tik Chiu, Xingqian Xu, Yunchao Wei, Zilong Huang, Alexander Schwing, Robert Brunner, Hrant Khachatrian, Hovnatan Karapetyan, Ivan Dozier, Greg Rose, David Wilson, Adrian Tudor, Naira Hovakimyan, Thomas S. Huang, and Honghui Shi. Agriculture-vision: A large aerial image database for agricultural pattern analysis. arXiv preprint arXiv:2001.01306, 2020. Available: https://arxiv.org/abs/2001.01306
