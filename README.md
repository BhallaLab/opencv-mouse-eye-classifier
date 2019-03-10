# About

This repository is fork of
[https://github.com/mrnugget/opencv-haar-classifier-trainin://github.com/mrnugget/opencv-haar-classifier-training]
. The details are on [this
blog](http://coding-robin.de/2013/07/22/train-your-own-opencv-haar-classifier.html).
Please visit this blog or the original repository for details.


# How to train?

Put negative images into the folder `./negative_images/` and positive images
into `./positive_images/`. And run

    make 

To use the existing images, you need to install `git-lfs` and run 
    
    git lfs pull
    make 

to pull the default images and start the process.


It will create a LBP classifier and save it to `./classifier_lpb/cascade.xml`.
To create a HAAR classifier (It will at least 100 times more time to train), run
`make haar` and it will save the classifier to `./classifier_haar/cascade.xml`.

# Testing

If you have your recording in TIFF file, you can use `test_cascade.py` file
e.g.,

    python ./test_cascade.py --tiff trial_008.tif --cascade ./trained_classifiers/mouse_eye.xml

And it will locate the trained pattern.

# Demo

[https://youtu.be/7dAdn_viR4c](https://youtu.be/7dAdn_viR4c)
