#!/bin/bash
# Run record_gps python script for microSWIFT 

#directories and files needed 
GPSwavesDir=/home/pi/microSWIFT/GPSwaves
config=/home/pi/microSWIFT/utils/Config.dat
utilsDir=/home/pi/microSWIFT/utils

#get PIDs  
#mswiftPID=$(ps -ef | grep "recordGPS.py" | grep -v grep | awk '{ print $2 }')
#echo "mswiftPID=" $mswiftPID

#add directories needed to run ecord and send gps app to pythonpath
export PYTHONPATH=$PYTHONPATH/$GPSwavesDir:/$utilsDir

#=================================================================================
#Run app
#=================================================================================

echo " --- RUN RECORD AND SEND GPS APP ---"
python3 recordGPS.py $config &
