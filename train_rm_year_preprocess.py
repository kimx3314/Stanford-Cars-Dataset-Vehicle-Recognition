import os
import numpy as np
import pandas as pd
import glob
import cv2

train_dir = os.listdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/train/')
train_csv = pd.read_csv('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/anno_train.csv', header = None)
data_labels = np.array(pd.read_csv('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/names.csv',\
                                   header = None))

idx = np.unique(np.array([label[0][0:-5] for label in data_labels]), return_index = True)[1]
labels_wo_year = np.array([np.array([label[0][0:-5] for label in data_labels])[idx] for idx in sorted(idx)])

pd.DataFrame(labels_wo_year).to_csv('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/wo_year_names.csv',\
                                    header = False, index = False)

os.chdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/train/')
for folder in train_dir:
    folder_read = folder + '/*.jpg'
    
    # reading from each class directory folder
    filelist = glob.glob(folder_read)
    all_files = [fname [-9:] for fname in filelist]
    sc_data = np.array([cv2.imread(fname) for fname in filelist])
    for i in range(len(sc_data)):
        file_dir = folder + '/' + all_files[i]
        annot = train_csv.loc[train_csv[0] == all_files[i]]
        x1 = annot[1][annot.index[0]]
        y1 = annot[2][annot.index[0]]
        x2 = annot[3][annot.index[0]]
        y2 = annot[4][annot.index[0]]
        cv2.imwrite(file_dir, sc_data[i][y1:y2, x1:x2])
