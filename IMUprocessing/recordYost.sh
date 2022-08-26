#!/bin/bash
# Run recordYost python script for microSWIFT 

#directories and files needed 
imuDir=/home/pi/microSWIFT/IMUprocessing
config=/home/pi/microSWIFT/utils/Config.dat
utilsDir=/home/pi/microSWIFT/utils

#get PIDs  
#mswiftPID=$(ps -ef | grep "recordYost.py" | grep -v grep | awk '{ print $2 }')
#echo "mswiftPID=" $mswiftPID

#add directories needed to run ecord and send gps app to pythonpath
export PYTHONPATH=$PYTHONPATH/$imuDir:/$utilsDir

#=================================================================================
#Run app
#=================================================================================

echo " --- RUN RECORD AND SEND IMU APP ---"
python3 recordYost.py $config &
