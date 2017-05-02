#!/usr/bin/env python
# encoding: utf-8

"""Read Data."""
import config
import handle_label

import os
import re
import random
import cv2
import numpy as np


class ImageReader():
    """Image reader."""

    def __init__(self, data_path, label_path, config):
        """Initializer."""
        self.data_path = data_path
        
        self.batch_size = config.batch_size
        self.class_num = config.class_num

        # buffer path
        self.buff_path = config.buff_path
        self.buff_tl_name = config.buff_tl_name

        # output image info
        self.img_width = config.img_height
        self.img_height = config.img_height

        # handle labels
        label_buff_path = os.path.join(self.buff_path, self.buff_tl_name)
        if not os.path.isfile(label_buff_path):
            with open(label_buff_path, 'w') as f:
                f.write(handle_label.handle(label_path, self.buff_path))

        self.records = []
        with open(label_buff_path) as f:
            for line in f:
                tmp = re.split(' *', line.strip())
                filename = tmp[0] + '.jpg'
                labels = [int(_) for _ in tmp[1:self.class_num+1]]
                self.records.append((filename, labels))

        self.total_size = len(self.records)

    def get_batch(self, distort=False, index=None):
        """Get a random batch of images if index == None."""
        if index is None:
            selected_ids = random.sample(xrange(self.total_size),
                                         self.batch_size)
        else:
            selected_ids = list(xrange(index, index + self.batch_size))
        img_list, label_list, name_list = [], [], []
        for _ in selected_ids:
            label_list.append(self.records[_][1])
            name_list.append(self.records[_][0])
            img = cv2.imread(os.path.join(self.data_path, self.records[_][0]))
            # should be replaced by chop, instead of resizing
            img = cv2.resize(img, (self.img_width, self.img_height))
            img_list.append(img)
        out_imgs = self._img_preprocess(np.stack(img_list))
        out_labels = self._label_preprocess(np.stack(label_list))
        return out_imgs, out_labels, name_list

    def _img_preprocess(self, imgs):
        output = np.reshape(imgs, [-1, self.img_height, self.img_width, 3])
        # TODO substract mean or more preprocess
        return output

    def _label_preprocess(self, label):
        output = np.reshape(label, [-1, self.class_num]).astype(np.float32)
        return output


def main():
    """Test Module."""
    train_config = config.Config()
    reader = ImageReader('../data/JPEGImages/', '../data/labels/',
                         train_config)
    imgs, labels, names = reader.get_batch(index=1)
    print(imgs.shape)
    print(names)
    print(labels[0])


if __name__ == '__main__':
    main()
