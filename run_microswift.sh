#!/bin/bash

# Run GPS and Yost bash files for microswift
cd /home/pi/microSWIFT/GPSWaves
bash recordGPS.sh &

cd /home/pi/microSWIFT/IMUprocessing
bash recordYost.sh &
