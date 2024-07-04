# all:
# 	g++ modelo.cpp --std=c++17 -I/usr/local/include/opencv4/ -L/home/paul/opencv/build/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_objdetect -lopencv_ml -o modelo

# run:
# 	./modelo


# all:
# 	g++ testing.cpp --std=c++17 -I/usr/local/include/opencv4/ -L/home/paul/opencv/build/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_objdetect -lopencv_ml -o testing.bin

# run:
# 	./testing.bin

all:
	g++ Principal.cpp --std=c++17 -I/usr/local/include/opencv4/ -L/home/paul/opencv/build/lib/ -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_videoio -lopencv_objdetect -lopencv_ml -o vision.bin

run:
	./vision.bin
