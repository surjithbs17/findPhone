import numpy as np
import os
from xml.etree import ElementTree
import pickle
import pandas as pd
import matplotlib.pyplot as plt

class CSV_preprocessor(object):
    
    def __init__(self,data_path="find_phone/"):
        self.path_prefix = data_path
        self.num_classes = 1
        self.data = dict()
        self.get_data()
    
    def populate_data(self,data):
        x,y,file = data["x"],data["y"],data["file"]
        filename = self.path_prefix +file
        img = plt.imread(filename)
        filename = filename
        #image_format = jpg
        width = int(x*img.shape[1])
        height = int(y*img.shape[0])
        offset_x = 30/img.shape[1]
        offset_y = 30/img.shape[0]
        xmaxs = [x+offset_x]
        xmins = [x-offset_x]
        ymaxs = [y+offset_y]
        ymins = [y-offset_y]
        class_name = 'Phone'
        bounding_box = [xmins,ymins,xmaxs,ymaxs]
        bounding_boxes.append(bounding_box)
        one_hot_class = to_one_hot(class_name) #add one hot
        one_hot_classes.append(one_hot_class)
        bounding_boxes = np.asarray(bounding_boxes)
        one_hot_classes = np.asarray(one_hot_classes)
        image_data = np.hstack((bounding_boxes, one_hot_classes))
        self.data[image_name] = image_data

    def one_hot_class(self,class_name):
        one_hot_vector = [0] * self.num_classes
        if name == 'Phone':
            one_hot_vector[0] = 1
        else:
            print('unknown label: %s' %name)

        return one_hot_vector

    def get_data(self):
        columns = ["file","x","y"]
        df = pd.read_csv(self.path_prefix+"labels.txt",header=None, delimiter=r"\s+",names=columns)
        #df = df.sort_values("file")
        for ind,data in df.iterrows():
            self.populate_data(data)

## example on how to use it
# import pickle
data = CSV_preprocessor('find_phone/').data
pickle.dump(data,open('phone_data.pkl','wb'))

