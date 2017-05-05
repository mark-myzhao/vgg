# VGG

Implement VGG model D (16 layers) with TensorFlow ...

## Project Directory Structure

- `./`: project root
  - `/script`: useful scripts
  - `/src`: source code
  - `/data`: training and testing dataset
    - `/JPEGImages`
    - `/labels`
  - `/label`: label for training
  - `/dest`: output (if any)
    - `/log`: log for tensorboard
    - `/param`: store check points
    - `/output`: output from the shell when model is training
    - `/result`: result generate by model testing
  - `/exchange`: exchange data between server and client
  - `/tmp`: store buffer files

## Dataset Structure

Source: [VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/)

- `VOC2012`
  - `/ImageSets`: labels of 20 classes
    - `*_train.txt`: used to train model
    - `*_val.txt`: used to validate and evaluate trained model
    - `*_trainval.txt`: `*_train.txt` + `*_val.txt`
  - `/JPEGImages`: 17125 raw colorful images

- ps. `.txt` format:
  - for valid images: `id_of_imgs  1` per line
  - for invalid images: `id_of_imgs -1` per line

## Run

- First, run:
```shell
chmod +x ./script/*
./script/init.sh
```

- Place your data and label in the right directory.
- Then you can run with `./script/run.sh`
- Or use tensorboard to check with `./script/check.sh`
