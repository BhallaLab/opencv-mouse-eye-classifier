"""test_cascade.py: 

Test OPENCV classifier.

"""
    
__author__           = "Dilawar Singh"
__copyright__        = "Copyright 2017-, Dilawar Singh"
__version__          = "1.0.0"
__maintainer__       = "Dilawar Singh"
__email__            = "dilawars@ncbs.res.in"
__status__           = "Development"

import tifffile
import cv2
import numpy as np

def main(args):
    cascade = cv2.CascadeClassifier(args.cascade)
    print( '[INFO] Processing %s' % args.tiff )
    frames = tifffile.imread(args.tiff)
    newFrames = []
    for fi, frame in enumerate( frames ):
        eyes = cascade.detectMultiScale(frame, scaleFactor=1.05, minNeighbors=20
                , minSize=(150,150)
                #  , maxSize=(250,250)
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
        newFrames.append(frame)
        cv2.imshow('Frame', frame)
        cv2.waitKey(10)

    # Save all the frames.
    tifffile.imsave('result.tif', np.array(newFrames))
    print( "[INFO ] Wrote new frames to results.tif" )
    
    

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
