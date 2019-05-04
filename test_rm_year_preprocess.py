import os
import numpy as np
import pandas as pd
import glob
import cv2

test_dir = os.listdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/test/')
test_csv = pd.read_csv('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/anno_test.csv', header = None)

os.chdir('C:/Users/sungi/Desktop/stanford-car-dataset-by-classes-folder/car_data/car_data/test/')
for folder in test_dir:
    folder_read = folder + '/*.jpg'
    
    # reading from each class directory folder
    filelist = glob.glob(folder_read)
    all_files = [fname [-9:] for fname in filelist]
    sc_data = np.array([cv2.imread(fname) for fname in filelist])
    for i in range(len(sc_data)):
        file_dir = folder + '/' + all_files[i]
        annot = test_csv.loc[test_csv[0] == all_files[i]]
        x1 = annot[1][annot.index[0]]
        y1 = annot[2][annot.index[0]]
        x2 = annot[3][annot.index[0]]
        y2 = annot[4][annot.index[0]]
        cv2.imwrite(file_dir, sc_data[i][y1:y2, x1:x2])

