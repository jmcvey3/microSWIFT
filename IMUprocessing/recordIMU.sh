#!/bin/bash
# Run record IMU for microSWIFT
#-------------------------------------------------------------------------------

#directories and files needed 
imuDir=home/pi/microSWIFT/IMUprocessing
config=/home/pi/microSWIFT/utils/Config.dat
utilsDir=home/pi/microSWIFT/utils

#get PIDs  
#imuPID=$(ps -ef | grep "recordIMU.py" | grep -v grep | awk '{ print $2 }')
#echo "imuPID=" $imuPID

#add directories needed to run imu app to pythonpath
export PYTHONPATH=$PYTHONPATH/$imuDir:/$utilsDir

#=================================================================================
#Run app
#=================================================================================

echo " --- RUN RECORD IMU APP ---"
python3 /home/pi/microSWIFT/IMUprocessing/recordIMU.py $config &
