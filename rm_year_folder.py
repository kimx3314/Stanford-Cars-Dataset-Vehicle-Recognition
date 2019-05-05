# created by Sean Sungil Kim                  < https://github.com/kimx3314 >
# used for removing the year level of the Stanford Cars Dataset

# meant to be used post merging, only removes the year level of the folder names
# change the directory according to your device before using



import os



os.chdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/train/')

train_dir = os.listdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/train/')
test_dir = os.listdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/test/')

for folder in train_dir:
    os.rename(folder, folder[:-5])

for folder in test_dir:
    os.rename(folder, folder[:-5])
