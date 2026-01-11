# Step 1: Compile the C++ wrapper to object file
g++ -c src/test_wrapper/opencv_wrapper.cpp -o opencv_wrapper.o \
    $(pkg-config --cflags opencv4) -fPIC

# Step 2: Compile the C main file to object file
gcc -c src/test_wrapper/grayscale_converter.c -o grayscale_converter.o \
    -I src/test_wrapper

# Step 3: Link everything together
g++ grayscale_converter.o opencv_wrapper.o -o grayscale_converter \
    $(pkg-config --libs opencv4)