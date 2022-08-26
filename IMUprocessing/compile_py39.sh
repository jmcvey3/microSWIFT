#
g++ -c -fPIC -c *.cpp  -ggdb -I/usr/include/python3.9 -L/usr/lib/python3.7 -I/usr/local/include/boost -L/usr/local/lib/boost -I/usr/local/include/boost/python -L/usr/local/lib/boost/python 

g++ -fPIC -shared -Wl,-soname,processIMU_lib.so -ggdb -o processIMU_lib.so  -Wno-undef -I/home/pi/microSWIFT/IMUprocessing *.o -I/usr/include/x86_64-linux-gnu -L/usr/lib/x86_64-linux-gnu -lm -g -I/usr/local/include/boost  -L/usr/local/lib/boost  -I/usr/include/python3.9 -I/usr/local/include/boost/python -L/usr/local/lib/boost/python -L/usr/local/lib/boost_python39 -L/usr/local/lib/boost_numpy39 -L/usr/lib/python3.9 -I/usr/include/python3.9m -lboost_python39 -lboost_numpy39


#g++ -c -fPIC hello.cpp -o hello.o
#g++ -shared -Wl,-soname,hello.so -o hello.so  hello.o -lpython3.7 -lboost_python



