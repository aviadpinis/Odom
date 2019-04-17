#!/usr/bin/env python
# coding: utf-8

import os
import re

class ReadData:
    def __init__(self, path):
        self.path = path
    #  For correct sorting.
    def atoi(self,text):
        return int(text) if text.isdigit() else text

    def natural_keys(self,text):
        '''
        alist.sort(key=natural_keys) sorts in human order
        http://nedbatchelder.com/blog/200712/human_sorting.html
        (See Toothy's implementation in the comments)
        '''
        return [self.atoi(c) for c in re.split('(\d+)', text)]

    path = '/home/aviad/Desktop/src/data/Images/odo360nodoor/odo360nodoor_orginal'

    def exportNameImages(self):
        imagesName = os.listdir(self.path)
        imagesName.sort(key=self.natural_keys)
        return [self.path + '/' + name for name in imagesName]
