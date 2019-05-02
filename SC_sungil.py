# created by Sean Sungil Kim                  < https://github.com/kimx3314 >
# used for loading the Stanford Cars images, bounding boxes, classes and labels        (load_images)
#          detecting the vehicle of interest within images utilizing bounding boxes    (vehicle_detect)
#          saving images as jpg files                                                  (save_as_jpg)
#          splitting Stanford Cars class labels into Make/Model/Type/Year              (split_labels)
#          saving the split class labels into a csv                                    (csv_convrt)
#          removing the year in the Stanford Cars labels and adjusting the class_nums  (rmv_year)
#          removing the year and model in the labels and adjusting the class_nums      (rmv_model_year)
#          converting images to grayscale                                              (gray_convrt)
#          obtaining average dimension size                                            (avg_size)
#          resizing images                                                             (resize_all)
#          reshaping images                                                            (clf_reshape)
#          converting images to edge detected images                                   (canny_edge_convrt)
#          computing histogram of gradients features                                   (hog_compute)
#          comparing image thresholding methods                                        (compare_thresh)
#          performing random under-sampling                                            (under_sample)



import numpy as np
import pandas as pd
import cv2
import os
import glob
import time
from scipy.io import loadmat
import matplotlib.pyplot as plt
from skimage.feature import hog



def load_images(img_dir, train_annot_dir, test_annot_dir, cls_dir):
    
    start_ts = time.time()
    os.chdir(img_dir)
    sc_data = np.array([cv2.imread(filename) for filename in sorted(glob.glob('*.jpg'), key = lambda x: int(os.path.splitext(x)[0]))])
    os.chdir('C:/Users/sungi/Documents/DSC672/')
    
    if img_dir == 'saved_images/training/' or img_dir == 'saved_images/resized/training/':
        meta = loadmat(cls_dir)
        cls_name = meta['class_names'][0]
        data_labels = np.array([img_label[0] for img_label in cls_name])

        train_mat = loadmat(train_annot_dir)
        train_annot = train_mat['annotations'][0]

        train_data_class = np.array([img_annot[4][0][0] for img_annot in train_annot])
        
        print("Training image loading runtime:", time.time()-start_ts)
    
        return sc_data, train_data_class, data_labels
    
    elif img_dir == 'saved_images/testing/' or img_dir == 'saved_images/resized/testing/':
        test_mat = loadmat(test_annot_dir)
        test_annot = test_mat['annotations'][0]

        test_data_class = np.array([img_annot[4][0][0] for img_annot in test_annot])
        
        print("Testing image loading runtime:", time.time()-start_ts)
    
        return sc_data, test_data_class
    
    elif img_dir == 'cars_train/cars_train/cars_train/':
        meta = loadmat(cls_dir)
        cls_name = meta['class_names'][0]
        data_labels = np.array([img_label[0] for img_label in cls_name])

        train_mat = loadmat(train_annot_dir)
        train_annot = train_mat['annotations'][0]

        train_data_class = np.array([img_annot[4][0][0] for img_annot in train_annot])
        
        x1 = np.array([img_annot[0][0][0] for img_annot in train_annot])
        y1 = np.array([img_annot[1][0][0] for img_annot in train_annot])
        x2 = np.array([img_annot[2][0][0] for img_annot in train_annot])
        y2 = np.array([img_annot[3][0][0] for img_annot in train_annot])

        bounding_boxes = []
        for i in range(len(x1)):
            bounding_boxes.append([x1[i], x2[i], y1[i], y2[i]])
        bounding_boxes = np.array(bounding_boxes)

        print("Training image loading runtime:", time.time()-start_ts)

        return sc_data, bounding_boxes, train_data_class, data_labels

    elif img_dir == 'cars_test/':
        test_mat = loadmat(test_annot_dir)
        test_annot = test_mat['annotations'][0]

        test_data_class = np.array([img_annot[4][0][0] for img_annot in test_annot])
        
        x1 = np.array([img_annot[0][0][0] for img_annot in test_annot])
        y1 = np.array([img_annot[1][0][0] for img_annot in test_annot])
        x2 = np.array([img_annot[2][0][0] for img_annot in test_annot])
        y2 = np.array([img_annot[3][0][0] for img_annot in test_annot])

        bounding_boxes = []
        for i in range(len(x1)):
            bounding_boxes.append([x1[i], x2[i], y1[i], y2[i]])
        bounding_boxes = np.array(bounding_boxes)

        print("Testing image loading runtime:", time.time()-start_ts)

        return sc_data, bounding_boxes, test_data_class



def vehicle_detect(sc_data, bounding_boxes):
    
    detected_sc = []
    for i in range(len(sc_data)):
        x1 = bounding_boxes[i][0]
        x2 = bounding_boxes[i][1]
        y1 = bounding_boxes[i][2]
        y2 = bounding_boxes[i][3]
        detected_sc.append(sc_data[i][y1:y2, x1:x2])
    
    return detected_sc



def save_as_jpg(train_img_data, test_img_data, save_dir = 'C:/Users/sungi/Documents/DSC672/saved_images/'):
    
    start_ts = time.time()
    os.chdir(save_dir)
    for i in range(len(train_img_data)):
        img_name = 'training/%s.jpg' % (str(i))
        cv2.imwrite(img_name, train_img_data[i])
        
    for i in range(len(test_img_data)):
        img_name = 'testing/%s.jpg' % (str(i))
        cv2.imwrite(img_name, test_img_data[i])

    os.chdir('C:/Users/sungi/Documents/DSC672/')
    print("Image saving runtime:", time.time()-start_ts)
    
    
def split_labels(input_label, wo_yr = 0, wo_mdyr = 0):
    
    make_List = []
    model_List = []
    year_List = []
    type_List = []
    
    if wo_yr == 0 and wo_mdyr == 0:
        for label in input_label:
            split_label = label.split(' ')

            if split_label[0] == 'Aston' or split_label[0] == 'Land' or split_label[0] == 'AM':
                make_List.append(' '.join(split_label[0:2]))
                model_List.append(' '.join(split_label[2:-2]))
                type_List.append(split_label[-2])
                year_List.append(split_label[-1])

            else:
                make_List.append(split_label[0])
                model_List.append(' '.join(split_label[1:-2]))
                type_List.append(split_label[-2])
                year_List.append(split_label[-1])

        return make_List, model_List, type_List, year_List
    
    elif wo_yr == 1:
        for label in input_label:
            split_label = label.split(' ')

            if split_label[0] == 'Aston' or split_label[0] == 'Land' or split_label[0] == 'AM':
                make_List.append(' '.join(split_label[0:2]))
                model_List.append(' '.join(split_label[2:-1]))
                type_List.append(split_label[-1])

            else:
                make_List.append(split_label[0])
                model_List.append(' '.join(split_label[1:-1]))
                type_List.append(split_label[-1])

        return make_List, model_List, type_List
    
    elif wo_mdyr == 1:
        for label in input_label:
            split_label = label.split(' ')

            if split_label[0] == 'Aston' or split_label[0] == 'Land' or split_label[0] == 'AM':
                make_List.append(' '.join(split_label[0:2]))
                type_List.append(split_label[-1])

            else:
                make_List.append(split_label[0])
                type_List.append(split_label[-1])

        return make_List, type_List



def csv_convrt(data_class, make_List, model_List, type_List, year_List, csv_name, wo_yr = 0, wo_mdyr = 0):
    
    df_List = []
    curr_List = []
    
    if wo_yr == 0 and wo_mdyr == 0:
        for cls in data_class:
            curr_List.append(cls)
            curr_List.append(make_List[cls-1])
            curr_List.append(model_List[cls-1])
            curr_List.append(type_List[cls-1])
            curr_List.append(year_List[cls-1])
            df_List.append(curr_List)
            curr_List = []
        df_Array = np.array(df_List)

        pd.DataFrame(df_Array).to_csv(csv_name, header = ['Class', 'Make', 'Model', 'Type', 'Year'], index = None)

        return pd.DataFrame(df_Array, columns = ['Class', 'Make', 'Model', 'Type', 'Year'])
        
    elif wo_yr == 1:
        for cls in data_class:
            curr_List.append(cls)
            curr_List.append(make_List[cls-1])
            curr_List.append(model_List[cls-1])
            curr_List.append(type_List[cls-1])
            df_List.append(curr_List)
            curr_List = []
        df_Array = np.array(df_List)

        pd.DataFrame(df_Array).to_csv(csv_name, header = ['Class', 'Make', 'Model', 'Type'], index = None)

        return pd.DataFrame(df_Array, columns = ['Class', 'Make', 'Model', 'Type'])
        
    elif wo_mdyr == 1:
        for cls in data_class:
            curr_List.append(cls)
            curr_List.append(make_List[cls-1])
            curr_List.append(model_List[cls-1])
            curr_List.append(type_List[cls-1])
            curr_List.append(year_List[cls-1])
            df_List.append(curr_List)
            curr_List = []
        df_Array = np.array(df_List)

        pd.DataFrame(df_Array).to_csv(csv_name, header = ['Class', 'Make', 'Type'], index = None)

        return pd.DataFrame(df_Array, columns = ['Class', 'Make', 'Type'])



def rmv_year(data_labels, all_class):
    
    idx = np.unique(np.array([label[0:-5] for label in data_labels]), return_index = True)[1]
    labels_wo_year = np.array([np.array([label[0:-5] for label in data_labels])[idx] for idx in sorted(idx)])
    original_cls_cnt = max(all_class)

    a = 0
    b = 2
    comb_List = []
    for i in range(len(idx)-1):
        comb_List.append(sorted(idx)[a:b])
        a += 1
        b += 1

    for i in range(original_cls_cnt):
        for j in range(len(comb_List)):
            if comb_List[j][0] <= i < comb_List[j][1]:
                class_idx = np.where(all_class == i+1)
                all_class[class_idx] = comb_List.index(comb_List[j])+1
                #print(i, comb_List.index(comb_List[j])+1)
            elif j == len(comb_List)-1:
                if i == comb_List[-1][1]:
                    class_idx = np.where(all_class == i+1)
                    all_class[class_idx] = comb_List.index(comb_List[j])+2
                    #print(i, comb_List.index(comb_List[j])+2)
        
    return labels_wo_year, all_class



def rmv_model_year(data_labels, all_class):
    
    labels_wo_make_year = []
    for label in data_labels:
        if label.split(' ')[0] == 'Aston' or label.split(' ')[0] == 'Land':
            labels_wo_make_year.append(' '.join(label.split(' ')[0:2]))
        else:
            labels_wo_make_year.append(label.split(' ')[0])

    idx = sorted(np.unique(labels_wo_make_year, return_index = True)[1])
    labels_wo_make_year = np.array(labels_wo_make_year)[idx]
    original_cls_cnt = max(all_class)

    a = 0
    b = 2
    comb_List = []
    for i in range(len(idx)-1):
        comb_List.append(idx[a:b])
        a += 1
        b += 1

    for i in range(original_cls_cnt):
        for j in range(len(comb_List)):
            if comb_List[j][0] <= i < comb_List[j][1]:
                class_idx = np.where(all_class == i+1)
                all_class[class_idx] = comb_List.index(comb_List[j])+1
                #print(i, comb_List.index(comb_List[j])+1)
            elif j == len(comb_List)-1:
                if i == comb_List[-1][1]:
                    class_idx = np.where(all_class == i+1)
                    all_class[class_idx] = comb_List.index(comb_List[j])+2
                    #print(i, comb_List.index(comb_List[j])+2)
    
    return labels_wo_make_year, all_class



def gray_convrt(input_data):

    # single rgb image input
    if len(input_data.shape) == 3:
        # converting input image to grayscale
        data_gray = np.array(cv2.cvtColor(input_data, cv2.COLOR_BGR2GRAY))
    
    # multiple rgb image inputs
    else:
        # converting input images to grayscale
        data_gray = np.array([cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in input_data])

    return data_gray



def avg_size(x_data):
    
    x_size_List = []
    y_size_List = []
    aspect_ratio_List = []
    for data in x_data:
        # get the minimum pixel length (images are not squares)
        x_size_List.append(data.shape[1])
        y_size_List.append(data.shape[0])
        aspect_ratio_List.append(data.shape[1]/data.shape[0])
    
    return (int(np.average(x_size_List)), int(np.average(y_size_List))), x_size_List, y_size_List, aspect_ratio_List



def resize_all(input_data, size = (50, 50)):

    start_ts = time.time()
    
    # resizing input images
    data_resized = np.array([cv2.resize(img, size) for img in input_data])

    print("Data resizing runtime:", time.time()-start_ts)
    
    return data_resized



def clf_reshape(input_data):
    
    # image flattening, reshaping the data to the (samples, feature) matrix format
    n_samples = len(input_data)
    data_reshaped = input_data.reshape((n_samples, -1))
    
    return data_reshaped



def canny_edge_convrt(input_data):

    # single rgb image input
    if len(input_data.shape) == 3:
        data_edge = cv2.Canny(input_data, 100, 200)
    
    # multiple rgb image inputs
    else:
        # converting input images to edge detected images
        data_edge = np.array([cv2.Canny(img, 100, 200) for img in input_data])

    return data_edge


    
def hog_compute(input_data):
    
    start_ts = time.time()
    # single rgb image input
    if len(input_data.shape) == 3:
        # computing HOG features of input images
        out, data_hog = hog(input_data, pixels_per_cell = (2, 2), visualize = True, multichannel = True)
    
    # multiple rgb image inputs
    else:
        # computing HOG features of input images
        hog_output = [hog(img, pixels_per_cell = (2, 2), visualize = True, multichannel = True) for img in input_data]
        data_hog = [hog_img for out, hog_img in hog_output]
    
    print("HOG feature computation runtime:", time.time()-start_ts)

    return data_hog



def compare_thresh(input_data):
    
    # rgb image input
    if len(input_data.shape) == 3:
        img = gray_convrt(input_data)
    
    fig = plt.figure(figsize = (12, 30))
    
    # global thresholding
    ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Adaptive Mean Thresholding
    th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)

    # Adaptive Gaussian Thresholding
    th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Otsu's thresholding
    ret4, th4 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    ret5, th5 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    titles = ['Global Thresholding (v = 127)', 'Adaptive Mean Thresholding', 'Adaptive Gaussian Thresholding', \
              "Otsu's Thresholding", "Otsu's Thresholding w/ Gaussian Filtering"]
    images = [th1, th2, th3, th4, th5]
    
    # plotting 5 different thresholding methods
    for i in range(5):
        plt.subplot(5, 1, i+1)
        plt.imshow(images[i], 'gray')
        plt.title(titles[i])
        plt.xticks([]),plt.yticks([])
    plt.show()



def under_sample(x_data, y_data):

    # obtaining class labels and their counts
    labels_arr, class_count = np.unique(y_data, return_counts = True)

    counter = 0
    # for each class label, performing random under-sampling using the indices
    for cls in labels_arr:
        get_indexes = lambda x, xs: [i for (y, i) in zip(xs, range(len(xs))) if x == y]
        cls_idx = get_indexes(cls, y_data)
        
        # obtaining all image instances for the class label "cls"
        reshaped_subset = x_data[cls_idx]
        class_subset = y_data[cls_idx]
        
        # obtaining the indices for the purpose of random sampling without replacement
        idx = np.random.choice(np.arange(len(reshaped_subset)), min(class_count), replace = False)
        
        if counter == 0:
            # applying the randomly sampled indices on the subsets
            x_und_smpl_data = reshaped_subset[idx]
            y_und_smpl_data = class_subset[idx]
            counter += 1
        else:
            # applying the randomly sampled indices on the subsets
            x_und_smpl_data = np.concatenate((x_und_smpl_data, reshaped_subset[idx]))
            y_und_smpl_data = np.concatenate((y_und_smpl_data, class_subset[idx]))
            counter += 1

    shuf_idx = np.random.permutation(len(x_und_smpl_data))
    x_und_smpl_data, y_und_smpl_data = x_und_smpl_data[shuf_idx], y_und_smpl_data[shuf_idx]
        
    return x_und_smpl_data, y_und_smpl_data
















