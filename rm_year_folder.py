import os

os.chdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/train/')

train_dir = os. listdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/train/')
test_dir = os. listdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/test/')

for folder in train_dir:
    os.rename(folder, folder[:-5])

for folder in test_dir:
    os.rename(folder, folder[:-5])
