# Raccoon
This project is detecting raccoons and localizing them using the PyTorch framework.

## Data
Data is from a public data set on roboflow, found [here.](https://public.roboflow.com/object-detection/raccoon/2)

The original data set has 195 images of with 212 labeled raccoons. I created a class to horizontally flip the image and bounding boxes across the center vertical line. After the images were flipped, they were added to the training set for a grand total of 390 images.

The data set was split into a train and test set, with a 60/40 split (234 in the train set and 156 in the test set).

Each image is resized to 3 x 300 x 400.

## Model
I used the pretrained Faster R-CNN ResNet 50 model from the torchvision library. I adapted the last layer to output two classes (raccoon or other).

## Results
After 25 epochs, I am get a mean IOU of 84% on the test set, as shown in the following graph:
![image](https://user-images.githubusercontent.com/26016287/123862549-191bcb00-d8ee-11eb-8683-ea0000c90fc6.png)

Upon viewing the results, the model sometimes struggles to identify if there are two or more raccoons in the same image. An improvement would be to gather more images of multiple raccoons or use additional synthetic data techniques.

I plan to add metrics for the precision, recall, and F1 score in the future.
