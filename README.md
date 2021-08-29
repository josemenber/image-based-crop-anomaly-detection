# Image-based crop anomaly detection using Artificial Intelligence techniques

This repository makes available the source code and public dataset for the review of work, "DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning", published with open access by Scientific Reports: https://www.nature.com/articles/s41598-018-38343-3. The DeepWeeds dataset consists of 17,509 images capturing eight different weed species native to Australia in situ with neighbouring flora. In this work, we present an improvement over the initial work. The dataset was classified to an average accuracy of 96.6 % with the ResNet50 deep convolutional neural network and an average accuracy of 95.6 % with the Inception-v3 model.

The contents of this repository are released under an [Apache 2 license](https://github.com/josemenber/image-based-crop-anomaly-detection/blob/master/LICENSE).

## Data organization

Images are assigned unique filenames that include the date/time the image was photographed and an ID number for the instrument which produced the image. The format is like so: `YYYYMMDD-HHMMSS-ID`, where the ID is simply an integer from 0 to 3. The unique filenames are strings of 17 characters, such as 20170320-093423-1.

## Labels

The `labels.csv` file assigns species labels to each image. It is a comma separated text file in the format:

```
Filename,Label,Species
...
20170207-154924-0,jpg,7,Snake weed
20170610-123859-1.jpg,1,Lantana
20180119-105722-1.jpg,8,Negative
...
```

## Download images

The images are available for download in the repository (images folder). You can also use the python script created to download automatically by executing the following command: 

```
    python3 download_images.py
```

This command will download and unzip a file containing the images.

## Download and evaluate models

We provide the most successful ResNet50 and InceptionV3 models saved in Keras hdf5 model format. Due to the size of the models they are hosted outside of the Github repository. The models must be downloaded [here](https://drive.google.com/file/d/1tGmL8fX2pdlkjepKWdfw_I1pFpRXnd2S/view?usp=sharing) (496.7MB). 

When downloading the file you have to unzip it inside the repository folder. Once unzipped you can test the models by running the command: 

```
    python3 eval_models.py
```

## Generate new anomaly images

To generate new anomaly images by applying transformations to the initial images (expansion of the training set) the Python script `generate_images.py` has been created. This script receives two parameters:

* `--noise`: to add gaussian noise to the images (default is not applied).
    
* `--num_imgs N`: where N is an integer indicating the number of images for each initial anomaly image. By default this value is 1.


For help on how to run this script execute the command:

```
    python3 generate_images.py -h
```

## Train and test models

`train_models.py` python script trains and evaluates Keras base implementation of ResNet50 and InceptionV3 on the DeepWeeds dataset, pre-trained with ImageNet weights.This script receives two parameters:

* `--noise`: to add gaussian noise to the images (default is not applied).

* `--augmentation`: to add the new images previously generated with the script *generate_imapges.py*  (default is not applied).

* `--model modelname`: to indicate the model name. *Modelname* takes the values: 'resnet-50' or 'inception-v3'.
    
* `--max_epochs N`: where N is an integer indicating the maximum number of epochs for training.

For help on how to run this script execute the command:

```
    python3 train_models.py -h
```


## Dependencies

The required Python packages to execute the scripts are listed in `requirements.txt` file.

## Citation

[1] IEEE style citation: “A. Olsen, D. A. Konovalov, B. Philippa, P. Ridd, J. C. Wood, J. Johns, W. Banks, B. Girgenti, O. Kenny, J. Whinney, B. Calvert, M. Rahimi Azghadi, and R. D. White, “DeepWeeds: A Multiclass Weed Species Image Dataset for Deep Learning,” *Scientific Reports*, vol. 9, no. 2058, **2** 2019. [Online]. Available: https://doi.org/10.1038/s41598-018-38343-3”

[2] Mang Tik Chiu, Xingqian Xu, Yunchao Wei, Zilong Huang, Alexander Schwing, Robert Brunner, Hrant Khachatrian, Hovnatan Karapetyan, Ivan Dozier, Greg Rose, David Wilson, Adrian Tudor, Naira Hovakimyan, Thomas S. Huang, and Honghui Shi. Agriculture-vision: A large aerial image database for agricultural pattern analysis. arXiv preprint arXiv:2001.01306, 2020.
