OPENCV=/usr/local/opencv3
OPENCV_LIBS=highgui imgcodecs core imgproc

ricochet: ricochet.cc
	g++ -g -O3 --std=c++11 -I${OPENCV}/include -L${OPENCV}/lib $? $(OPENCV_LIBS:%=-lopencv_%) -o $@
