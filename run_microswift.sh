#!/bin/bash

LOGFILE=/home/pi/microSWIFT/microswift.log

(
    # Run GPS and Yost bash files for microswift
    #directories and files needed
    wavesDir=/home/pi/microSWIFT/GPSwaves
    imuDir=/home/pi/microSWIFT/IMUprocessing
    config=/home/pi/microSWIFT/utils/Config.dat
    utilsDir=/home/pi/microSWIFT/utils

    #add directories needed to run ecord and send gps app to pythonpath
    export PYTHONPATH=$PYTHONPATH/$wavesDir:/$imuDir:/$utilsDir

    #=================================================================================
    #Run app
    #=================================================================================

    echo " --- RUN RECORD AND SEND GPS APP ---"
    python3 GPSwaves/recordGPS.py $config &

    echo " --- RUN RECORD AND SEND IMU APP ---"
    python3 IMUprocessing/recordIMU.py $config &
) >& $LOGFILE