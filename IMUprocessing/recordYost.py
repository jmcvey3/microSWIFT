#! /usr/bin/python3

#standard imports 
import time, os, sys
from datetime import datetime, timedelta
import numpy as np
import serial
from logging import *
import csv
import threading

#my imports 
from config3 import Config


#---------------------------------------------------------------
configDat = sys.argv[1]
configFilename = configDat #Load config file/parameters needed

config = Config() # Create object and load file
ok = config.loadFile( configFilename )
if( not ok ):
    sys.exit(0)

#set up logging
logDir = config.getString('Loggers', 'logDir')
LOG_LEVEL = config.getString('Loggers', 'DefaultLogLevel')
#format log messages (example: 2020-11-23 14:31:00,578, recordIMU - info - this is a log message)
#NOTE: TIME IS SYSTEM TIME
LOG_FORMAT = ('%(asctime)s, %(filename)s - [%(levelname)s] - %(message)s')
#log file name (example: home/pi/microSWIFT/recordIMU_23Nov2020.log)
LOG_FILE = (logDir + '/' + 'recordIMU' + '_' + datetime.strftime(datetime.now(), '%d%b%Y') + '.log')
logger = getLogger('system_logger')
logger.setLevel(LOG_LEVEL)
logFileHandler = FileHandler(LOG_FILE)
logFileHandler.setLevel(LOG_LEVEL)
logFileHandler.setFormatter(Formatter(LOG_FORMAT))
logger.addHandler(logFileHandler)

#load parameters from Config.dat
#system parameters 
floatID = os.uname()[1]
#floatID = config.getString('System', 'floatID')

dataDir = config.getString('System', 'dataDir')
burst_interval=config.getInt('System', 'burst_interval')
burst_time=config.getInt('System', 'burst_time')
burst_seconds=config.getInt('System', 'burst_seconds')

bad = config.getInt('System', 'badValue')

#IMU parameters---------------------------------------------
# Will need to update config for correct gpio and freq
imu_port = config.getString('IMU', 'port')
imuFreq = config.getFloat('IMU', 'imuFreq')
imu_samples = imuFreq*burst_seconds
imu_fs_period = int(1/imuFreq * 1e6) # sampling period in microseconds

#------------------------------------------------------------
#Yost IMU setup
# open serial port
ser = serial.Serial(port=imu_port, baudrate=115200)
# serial commands
ser.write(':86\r\n'.encode('ascii')) # stop running
time.sleep(1)
ser.write(':80,0,38,39,40,45,255,255,255\r\n'.encode('ascii')) # sensor config
time.sleep(0.1)
ser.write((':82,{},-1,0\r\n'.format(imu_fs_period)).encode('ascii')) # timing
time.sleep(0.1)
# Sanity check on sampling rate
ser.write(':83\r\n'.encode('ascii')) # get timing
time.sleep(0.1)
assert ser.in_waiting==13
timing = ser.read(13).decode()
assert int(timing[0:6])==imu_fs_period
time.sleep(5)

# There appears to be a time delay between setting the sampling rate 
# and the sampling rate actually being set - starting and stopping IMU 
# w/i while loop
#---------------------------------------------------------------
 

# Main loop will read the acceleration and magnetometer values every second
# and print them out.
imu = []
isample = 0
tStart = time.time()
#-------------------------------------------------------------------------------
#LOOP BEGINS
#-------------------------------------------------------------------------------
logger.info('---------------recordIMU.py------------------')
while True:
    ser.write(':85\r\n'.encode('ascii')) # start IMU
    now=datetime.utcnow()
    if now.minute == burst_time or now.minute % burst_interval == 0 and now.second == 0:
        logger.info('Starting burst')
        
        #create new file for new burst interval 
        fname = dataDir + floatID + '_IMU_'+'{:%d%b%Y_%H%M%SUTC.dat}'.format(datetime.utcnow())
        logger.info('File name: %s' %fname)
        
        with open(fname, 'w', newline='\n') as imu_out:
            logger.info('Open file for writing: %s' %fname)
            t_end = time.time() + burst_seconds #get end time for burst
            isample=0
            
            while time.time() <= t_end or isample < imu_samples:
                try:
                    # Get data
                    timestamp='{:%Y-%m-%d %H:%M:%S.%f}'.format(datetime.utcnow())
                    data0 = ser.readline().decode()
                    data38 = ser.readline().decode()
                    data39 = ser.readline().decode()
                    data40 = ser.readline().decode()
                    data45 = ser.readline().decode()
 
                except Exception as e:
                    logger.info(e)
                    logger.info("Error reading IMU data")

                gyro_x, gyro_y, gyro_z = [float(x) for x in data38.split(',')]
                accel_x, accel_y, accel_z = [float(x)*9.81 for x in data39.split(',')] # "G" to m/s^2
                mag_x, mag_y, mag_z = [float(x)*100 for x in data40.split(',')] # gauss to uT
                q_x, q_y, q_z, q_w = [float(x) for x in data0.split(',')]
                conf = float(data45)

                imu_out.write('%s,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f\n' %(timestamp,accel_x,accel_y,accel_z,gyro_x,gyro_y,gyro_z,mag_x,mag_y,mag_z,q_x,q_y,q_z,q_w,conf))

                isample = isample + 1

                if time.time() >= t_end and 0 < imu_samples-isample <= 40:
                    continue
                elif time.time() > t_end and imu_samples-isample > 40:
                    break

            ser.write(':86\r\n'.encode('ascii')) # stop running
            logger.info('End burst')
            logger.info('IMU samples %s' %isample)  

        time.sleep(0.1)
