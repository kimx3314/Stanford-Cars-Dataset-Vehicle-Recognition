# Advanced Machine Learning (CAPSTONE PROJECT)

### Stanford Cars Dataset - Vehicle Recognition
using Python

by Sean Sungil Kim

  The public Stanford Cars Dataset (which can be downloaded here: http://ai.stanford.edu/~jkrause/cars/car_dataset.html) contains total 16,185 images of cars. There are a total of 196 classes of cars in this original dataset. The data is split in half to be used as training and testing sets. The data also comes with class labels and bounding boxes for all images. The classes are typically at the level of Make, Model and Year (e.g. Tesla Model S 2012 or BMW M3 coupe 2012). The sizes of each image are different. Utilization of the bounding boxes is essential in the pre-processing phase to first obtain images that focus on the objects of interest, which in this case are the vehicles. The class labels may be split and re-engineered in order to further reduce total class count to simplify the recognition task. The actual images are in JPG format, but the data comes zipped in TGZ/TAR format.

  The Stanford Car Dataset will be utilized to build a vehicle recognition predictive model. The ultimate goal of the model is to classify a car’s year, make and model given an input image. This model could be further developed to be used in creating a mobile application that assists users in identifying cars of interest. The users would simply take a picture of the vehicle of interest and the application would return information (make, model and year) regarding the recognized vehicle. The users could also input a picture found on the Internet. Partnerships with other car dealership websites could be beneficial in enhancing the application quality, since the recognized vehicle name would be used in searching the partners’ database to obtain valuable information such as availability, price and so on. An improved model would result in direct reviews/subscription profit. This application could help people who are not familiar with cars or who simply want quick information without searching the Internet themselves. Another potential development idea of this project would be for traffic law enforcement. Traffic AI is a huge market globally. One example of this would be the China Transinfo Technology Corp. They focus on extracting features of vehicles the moment they appear in security cameras, which can assist the police to track down the targeted cars.

  Different classification algorithms such as RandomForest, SVM, some of the boosting approaches, in addition to Convolution Neural Networks will be explored. Both custom and state-of-the-art CNN architectures will be analyzed. Some of the feature extraction and feature selection methods will be explored as well, for the non-deep learning classifiers. The values of these models will be quantified in terms of performance and cost. Furthermore, in order to evaluate the true performance of the model, 30 images or more will be added to the validation set to gauge the true real-world predictive power.
