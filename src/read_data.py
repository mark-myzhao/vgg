#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import sys
import os
import re
import random
import math
import cv2

class ImageReader():
    def __init__(self, data_path, config):
	self.data_path = data_path
        self.batch_size = config.batch_size
	self.line_idx = 0 #Begin part

        if config.is_color:
            self.color_mode = 1
        else:
            self.color_mode = 0

    def batch(self, line_idx=None):
        filename_list = list()

	filename_list.append("1.jpg")#先测试能否读三张图片
	filename_list.append("2.jpg")
	filename_list.append("3.jpg")

        img_list = list()
        for filename in filename_list:
            img = cv2.imread(os.path.join(self.data_path, filename),
                    self.color_mode)
            #img = cv2.resize(img, (self.img_width, self.img_height))
            img_list.append(img)

	labels = [1,1,-1]#三张图片

        return (imgs.reshape([self.batch_size, self.img_height, self.img_width, 3]),
                labels.reshape([self.batch_size, 2]),
                filename_list)

def main():
	config.batch_size = 3
	config.is_color = 1
	r = ImageReader("/tmp", config)
	imgs, labels, filename_list = r.batch(0)
	
if __name__ == '__main__':
	main()
        

