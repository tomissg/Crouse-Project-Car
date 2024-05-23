This is a beta project for my masters course. The task of the project is to augment images of cars from VisData2019 in order to simulate photos picture taken in snowing day and ryn to find cars using yolov9.
The pipeline goes as followed 
1) I take the pictures and keep only those who have cars in them
2) Also I delete all the other boxes containing any other objects other than cars
3) I reshape the images to a fixed size of (different tries) while keeping track of the scaling factor in order to scale the annotations accordingly
4) When all that is done I create the needed format of the directories in order to creat the yaml file of the data in order to feed the pretrained yolov9 model

For this task is important to install all the necessary cuda packages based on your graphic card

Dataset: https://github.com/VisDrone/VisDrone-Dataset
Dataset Discription: https://github.com/VisDrone/VisDrone2018-DET-toolkit

Refference paper: Nordic Vehicle Dataset (NVD): Performance of Vehicle Detectors Using Newly Captured NVD From UAV in Different Snowy Weather Conditions
Yolov9 info: https://docs.ultralytics.com/models/yolov9/#citations-and-acknowledgements
