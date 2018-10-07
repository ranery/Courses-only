# -*- coding: utf-8 -*-
"""
@author : Haoran You

"""

import numpy as np
import struct

def read_image_files(filename, num):
    bin_file = open(filename, 'rb')
    buf = bin_file.read()
    index = 0
    # 前四个32位integer为以下参数
    # >IIII 表示使用大端法读取
    magic, numImage, numRows, numCols = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    image_sets = []
    for i in range(num):
        images = struct.unpack_from('>784B', buf, index)
        index += struct.calcsize('>784B')
        images = np.array(images)
        images = images/255.0
        images = images.tolist()
        # if i == 6:
        #     print ','.join(['%s'%x for x in images])
        image_sets.append(images)
    bin_file.close()
    return image_sets


def read_label_files(filename):
    bin_file = open(filename, 'rb')
    buf = bin_file.read()
    index = 0
    magic, nums = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')
    labels = struct.unpack_from('>%sB'%nums, buf, index)
    bin_file.close()
    labels = np.array(labels)
    return labels

def fetch_traingset(path):
    image_file = path + '/train-images.idx3-ubyte'
    label_file = path + '/train-labels.idx1-ubyte'
    images = read_image_files(image_file,60000)
    labels = read_label_files(label_file)
    return {'images': images,
            'labels': labels}

def fetch_testingset(path):
    image_file = path + '/t10k-images.idx3-ubyte'
    label_file = path + '/t10k-labels.idx1-ubyte'
    images = read_image_files(image_file,10000)
    labels = read_label_files(label_file)
    return {'images': images,
            'labels': labels}

def loaddata():
    # path = "./MNIST"
    path = "C:\\Users\dell-pc\Desktop\大四上\Computer_Vision\CNN\MNIST"
    train_data = fetch_traingset(path)
    test_data = fetch_testingset(path)
    return train_data, test_data

