### Assignment: Applied Machine Learning in the Cloud Final Project

### Authors: Atharv Vanarase (avv2116), Nathan Koenig (nak2172), Sophie Johnson (smj2173)

This project contains the code for training, evaluating, and inferencing fine-tuned ResNet-18 models against ImageNet, DAmageNet, and combined ImageNet and DAmageNet datasets. A subset of ImageNet/DAmageNet is used, looking at 10 image classes (dog, fish, golf ball etc).

## How to run train.py

The python script can be run by running the following command: 

``` python train.py ```

By default, we train on just the imagenette dataset with 10 epochs and a batch size of 32 images per batch.. By using the ```-d``` flag, we swap to training on the combined dataset. By using the ```-n``` flag, we can set a specific number of epochs to run. Finally, by setting the ```-b``` flag, we can choose a different batch size. The following shows an example run where we train on both ImageNet and DAmageNet for 100 epochs with a batch size of 32.

``` python train.py -d -n 100 -b 16 ```

## How to run accuracy.py

The python script can be run by running the following command:

``` python accuracy.py ```

By default, we choose the combined model and the combined test dataset to run the tests on. By using the ```-m``` flag, we can choose another model type and by setting the ```-t``` flag, we can choose another test dataset (options are **combined, damagenet, and imagenette**). The following shows an example run.

``` python accuracy.py -m ./outputs/imagenette_model.pth -t damagenet ```

## How to run server.py

The python script can be run by running the following command:

``` python server.py ```

From there, a Flask application will load up and the inferencing engine will be available at the following URL: http://localhost:5000/. **Note that you must have an NVIDIA GPU in order to run this script appropriately.**

## File Directory

The current file directory contains the code and report for the Emotion Recognition HW Assignment. The folder contains the following items:

1. train.py: This script is used to train the 2 different models. The script has adjustable parameters to allow us to either included DAmageNet data or not, choose a specific number of epochs, and set a specific batch size. It saves the trained model parameters and all accuracy and loss plots to the outputs directory.

2. accuracy.py: This script is used to check the accuracy of both of the previously trained models. The script has adjustable parameters to choose the model filepath and to choose the test dataset to use.

3. server.py: This script is used to run the webserver to allow for on-demand inferencing on our previously created models. 

4. Data Splitter.ipynb: This ipython notebook script was used previously to generate the final_combined, final_damagenet, and final_imagenette folders. The script no longer needs to be ran because all of the data is already in the github in the correct way, but it is added for visibility. In the code, we can see that the data is randomly split to be 75% training, 15% validation, and 10% test data for all 3 datasets.

5. final_imagenette folder: This folder contains the training, validation, and test datasets for the 10 ImageNet classes used for our ImageNet classification model. The training and validation sets are directly referenced in our train.py script, and the test set is directly referenced in the accuracy.py script.

6. final_damagenet folder: This folder contains the training, validation, and test datasets for the 10 DAmageNet classes used for our classification model. The test set is directly referenced in the accuracy.py script.

7. final_combined folder: This folder is a combination of the final_imagenette folder and the final_damagenet folder, and used to create our Combined Imagenet and DAmageNet model. The training and validation sets are directly referenced in our train.py script, and the test set is directly referenced in the accuracy.py script.

8. /templates folder: This folder contains all the HTML elements and templates used by our server.py script.

9. /outputs folder: This folder contains both of the models and all plots created during the previous train.py script runs. During the training process in the Google Cloud VM, we saved the outputs of the training (the model and the accuracy/loss plots) to this folder.

10. README.md: This markdown file contains all the necessary information about the project folder.

