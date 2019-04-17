import numpy as np
import cv2
import os
import re

#  For correct sorting.
def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [atoi(c) for c in re.split('(\d+)', text)]

path = '/home/aviad/Desktop/src/data/Images/rgbd_dataset_freiburg3_structure_notexture_far/rgb'

images = os.listdir(path)
images.sort(key=natural_keys)

prevImg = cv2.imread(path+'/'+images[0],0)
nextImg = cv2.imread(path+'/'+images[1],0)

sift = cv2.xfeatures2d.