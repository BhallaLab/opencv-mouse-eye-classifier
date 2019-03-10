"""test_cascade.py: 

Test OPENCV classifier.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import sys
import os
import pathlib
import numpy as np
from libtiff import TIFF
import cv2

def main(args):
    tifffile = args.tiff
    cascade = cv2.CascadeClassifier(args.cascade)
    print( '[INFO] Processing %s' % tifffile )
    tf = TIFF.open(tifffile)
    frames = tf.iter_images( )
    for fi, frame in enumerate( frames ):
        eyes = cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=10
                , minSize=(150,150), maxSize=(350,250)
                )
        if len(eyes)<1:
            continue
        # sort according to area.
        eyeWithArea = sorted([(x, x[-1]*x[-2]) for x in eyes], key=lambda x: x[-1])

        # Draw only the last one.
        for (ex,ey,ew,eh), ar in eyeWithArea[-1:]:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),255,2)
            #  roi = frame[ey:ey+eh,ex:ex+ew]
            #  cv2.imshow('ROI', roi)

        #  cv2.imshow('Frame', frame)
        cv2.waitKey(10)

if __name__ == '__main__':
    import argparse
    # Argument parser.
    description = '''Process TIFF file to locate eye.'''
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--tiff', '-t'
        , required = True
        , help = 'Input TIFF file'
        )
    parser.add_argument('--cascade', '-c'
        , required = True
        , help = 'LBP classifier (XML file)'
        )
    class Args: pass 
    args = Args()
    parser.parse_args(namespace=args)
    main(args)
