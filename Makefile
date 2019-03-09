all : lpb

OPTIONS:= -vec samples.vec -bg negatives.txt \
	    -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 1000\
	   -numNeg 600 -w 60 -h 40 -mode ALL \
	   -precalcValBufSize 1024\
	   -precalcIdxBufSize 1024 \

prepare:
	find ./positive_images -iname "*.png" > positives.txt
	find ./negative_images -iname "*.jpg" > negatives.txt
	perl bin/createsamples.pl positives.txt negatives.txt samples 1500\
	   "opencv_createsamples -bgcolor 0 -bgthresh 0 -maxxangle 1.1\
	   -maxyangle 1.1 maxzangle 0.5 -maxidev 40 -w 60 -h 40"
	python3 ./tools/mergevec.py -v samples/ -o samples.vec

haar:  prepare
	@echo "Generating HAAR cascade"
	mkdir -p classifier_haar
	opencv_traincascade -data classifier_haar $(OPTIONS)

lpb: prepare
	@echo "Generating LPB cascade"
	mkdir -p classifier_lpb
	opencv_traincascade -data classifier_lpb $(OPTIONS) -featureType LBP

