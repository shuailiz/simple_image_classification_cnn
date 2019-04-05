#!/bin/usr/python

import os
import cv2
import glob
from sklearn.utils import shuffle
import numpy as np

class DataProcesser(object):

    def __init__(self, class_names, image_size):
        self.class_names = class_names
        self.image_size = image_size
        self.total_num_data_points = 0
        self.data_loaded = False
        self.images = []
        self.labels = []
        self.cls = []
        self.image_names = []
        self.training_data = []
        self.validation_data = []

        self.training_data_start_ind = 0
        self.training_data_end_ind = 0
        self.valid_data_start_ind = 0
        self.valid_data_start_ind = 0
        self.training_ind_in_epoch = 0
        self.valid_ind_in_epoch = 0

    def get_training_data_size(self):
        ''' Get the total number of points in the training data set
        '''
        return self.training_data_end_ind - self.training_data_start_ind + 1

    def read_data_from_folder(self, folder_path):
        ''' Read data from a folder
        param:
        folder_path (str): path to the folder
        return:
        input_data: the data read from the folder
        '''
        for class_name in self.class_names:
            data_path = os.path.join(folder_path, class_name, '*g')
            data_files = glob.glob(data_path)
            for data_file in data_files:
                image = cv2.imread(data_file)
                # resize the image to be more reasonable
                image = cv2.resize(image, (self.image_size, self.image_size), 0, 0, cv2.INTER_LINEAR)
                # cast pixel value to be float
                image = image.astype(np.float32)
                # convert the pixle values to be between 0 and 1
                image = np.multiply(image, 1.0/255.0)
                label = np.zeros(len(self.class_names))
                label[self.class_names.index(class_name)] = 1.0
                cl = class_name
                self.images.append(image)
                self.labels.append(label)
                self.cls.append(cl)
                self.image_names.append(os.path.basename(data_file))

        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        self.image_names = np.array(self.image_names)
        self.cls = np.array(self.cls)
        self.images, self.labels, self.cls, self.image_names = shuffle(self.images, self.labels, self.cls, self.image_names)
        self.total_num_data_points = len(self.cls)
        self.data_loaded = True

    def read_data_from_hdf5(self, hdf5_file):
        ''' Read data from a hdf5 file
        param:
        hdf5_file: The hdf5 file handle
        return:
        input_data: The data read from the hdf5 file
        '''
        pass

    def read_data_from_file(self, file_handle, data_type):
        ''' Read data from a file
        param:
        file_handle: a file reader handle
        data_type (enum): the type of the data to read
        return:
        input_data: the data read from the file handle
        '''
        pass

    def save_data_to_hdf5(self, data, hadf5_file):
        ''' Save data to a hdf5 file
        param:
        data: The data to be saved
        hdf5_file: The file handle to the hdf5 file
        '''
        pass

    def validate_data_point(self, data_point):
        ''' Validate a data point 
        param:
        data_point: the data point to validate
        '''
        if data_point.shape[0] != self.image_size:
            print("expected first dim %d, given dim %d".format(self.image_size, data_point.shape[0]))
            return False

        if data_point.shape[1] != self.image_size:
            print("expected second dim %d, given dim %d".format(self.image_size, data_point.shape[1]))
            return False

        if np.isnan(data_point).any():
            print("None value found in the data point array")
            return False

        return True

    def divide_training_valid_data(self, training_size):
        ''' divide the data into training and validation for tensor flow network
        param:
        training_size (double): The the size of the training set in the whole data set
        return:
        training_data: The data for training
        '''
        training_data_num = int(training_size * self.total_num_data_points)
        self.training_data_start_ind = 0
        self.training_data_end_ind = training_data_num - 1
        self.valid_data_start_ind = training_data_num
        self.valid_data_end_ind = self.total_num_data_points - 1
        self.training_ind_in_epoch = self.training_data_start_ind
        self.valid_ind_in_epoch = self.valid_data_start_ind

    def next_training_batch(self, batch_size):
        ''' Generate the next batch of data for training
        param:
        batch_size: How large the batch will be
        return:
        training_batch: The batch of training data
        '''
        start = self.training_ind_in_epoch
        end = self.training_ind_in_epoch + batch_size - 1

        if start >= self.training_data_end_ind:
            start = self.training_data_start_ind
            end = self.training_data_start_ind + batch_size - 1
        elif end > self.training_data_end_ind:
            end = self.training_data_end_ind

        self.training_ind_in_epoch = end + 1
        return self.images[start:end], self.labels[start:end]

    def next_validate_batch(self, batch_size):
        ''' Generate the next batch of data for validation
        param:
        batch_size: How large the batch will be
        return:
        validation_batch: The batch of validation data
        '''
        start = self.valid_ind_in_epoch
        end = self.valid_ind_in_epoch + batch_size - 1

        if start >= self.valid_data_end_ind:
            start = self.valid_data_start_ind
            end = self.valid_data_start_ind + batch_size - 1
        elif end > self.valid_data_end_ind:
            end = self.valid_data_end_ind

        return self.images[start:end], self.labels[start:end]
