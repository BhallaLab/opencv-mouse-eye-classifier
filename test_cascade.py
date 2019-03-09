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
import matplotlib.pyplot as plt
import numpy as np
from libtiff import TIFF
import cv2

def main(  ):
    print( sys.argv )
    tifffile = sys.argv[1]
    cascade = cv2.CascadeClassifier(sys.argv[2])
    print( '[INFO] Processing %s' % tifffile )
    tf = TIFF.open( tifffile )
    frames = tf.iter_images( )
    for fi, frame in enumerate( frames ):
        eyes = cascade.detectMultiScale(frame)
        if len(eyes)<1:
            continue
        # sort according to area.
        eyeWithArea = sorted([(x, x[-1]*x[-2]) for x in eyes], key=lambda x: x[-1])
        for (ex,ey,ew,eh), ar in eyeWithArea[-1:]:
            cv2.rectangle(frame,(ex,ey),(ex+ew,ey+eh),255,2)
        cv2.imshow('Frame', frame)
        cv2.waitKey(10)

if __name__ == '__main__':
    main()
