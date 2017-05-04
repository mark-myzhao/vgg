"""Collect labels to generate a single label file."""
import re
import os


def handle(label_path, tmp_path):
    """Collect Labels."""
    class_name = [
        'aeroplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat',
        'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant',
        'sheep', 'sofa', 'train', 'tvmonitor'
        ]

    onehot = []
    with open(os.path.join(label_path, 'train.txt')) as f:
        for line in f:
            onehot.append(line.strip())

    for name in class_name:
        filename = name + '_train.txt'
        with open(os.path.join(label_path, filename)) as f:
            cur = 0
            for line in f:
                nd = '0' if re.split(' *', line.strip())[-1] == '-1' else '1'
                onehot[cur] += ' ' + nd
                cur += 1

    with open(os.path.join(tmp_path, 'label_train.txt'), 'w') as f:
        for line in onehot:
            f.write(line + '\n')


def main():
    """Test Module."""
    handle('../data/labels/', '../tmp/')


if __name__ == '__main__':
    main()
