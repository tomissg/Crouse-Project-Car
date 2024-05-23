This is a beta project for my masters course. The task of the project is to augment images of cars from VisDrone2019 in order to simulate photos picture taken in snowing day and ryn to find cars using yolov9.
The pipeline goes as followed 
1) I take the pictures and keep only those who have cars in them
2) Also I delete all the other boxes containing any other objects other than cars
3) I reshape the images to a fixed size of (different tries) while keeping track of the scaling factor in order to scale the annotations accordingly
4) When all that is done I create the needed format of the directories in order to creat the yaml file of the data in order to feed the pretrained yolov9 model

Problems need to be optimised:

The dataset was full of pictures for different kinds of object detection. My aproach was to delete ll other objects from the annotations and keep cars of a specific box size in order to avoid aving bland spots where a car was small in the picture but after the augmentation process might be a small dot or something similar meaninng the info of the shape of a car might be losed. Also eadge detection applied to try and make them mor destinct. One other problem was the inclusion of wrong labels in some annotations some images for example have spots of trees classified as car in the original data. For better results inspection of the images needed to manually choose the correct data. 

For this task is important to install all the necessary cuda packages based on your graphic card

Dataset: https://github.com/VisDrone/VisDrone-Dataset
Dataset Discription: https://github.com/VisDrone/VisDrone2018-DET-toolkit

Refference paper: Nordic Vehicle Dataset (NVD): Performance of Vehicle Detectors Using Newly Captured NVD From UAV in Different Snowy Weather Conditions
Yolov9 info: https://docs.ultralytics.com/models/yolov9/#citations-and-acknowledgements

Disclaimer: The aim of this project is implementing yolov9 and an introduction of object detection using yolov9. Optimization was not the main purpose of this project. 
