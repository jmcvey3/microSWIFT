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
import processIMU_lib

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
 

def process_imu_thread(filename, logger, imuFreq):
    #filename = "../data/raspberrypi_IMU_19Mar2022_042500UTC.dat"
    with open(filename, 'r') as fp:
        data = list(csv.reader(fp, delimiter=','))
    IMUdata = np.array(data)[:,1:-5].astype(float) # grab all data except time, quaternions and check

    # Yost defines y and z axes incorrectly (doesn't follow right hand rule)
    # processIMU_lib detrends these
    axs = IMUdata[:,0]  # accelerometer
    ays = IMUdata[:,2]
    azs = IMUdata[:,1]
    gxs = IMUdata[:,3]  # gyro
    gys = IMUdata[:,5]
    gzs = IMUdata[:,4]
    mxs = IMUdata[:,6] # magnetometer
    mys = IMUdata[:,8]
    mzs = IMUdata[:,7]

    mxo = np.double(0.) # magn calibration data (Yost self-calibrates)
    myo = np.double(0.) 
    mzo = np.double(0.)  
    Wd = np.double(0.) 
    fs = np.double(imuFreq) # sampling frequency

    nv=np.size(axs)

    # call processIMU
    # Requires at least 2048 data points at 4 Hz apparently, so at least 8:42 of data
    IMU_results = processIMU_lib.main_processIMU(nv, axs, ays, azs, gxs, gys, gzs, 
                                                    mxs, mys, mzs, mxo, myo, mzo, Wd, fs)
                                                    
    Hs = IMU_results[0] # significant wave height
    Tp = IMU_results[1] # peak wave period
    Dp = IMU_results[2] # peak wave direction
    E = np.squeeze(IMU_results[3]) # wave spectra energy
    f   = np.squeeze(IMU_results[4]) # wave spectra frequency
    a1 = np.squeeze(IMU_results[5]) # wave spectra a1 moment
    b1 = np.squeeze(IMU_results[6]) # wave spectra b1 moment
    a2 = np.squeeze(IMU_results[7]) # wave spectra a2 moment
    b2 = np.squeeze(IMU_results[8]) # wave spectra b2 moment
    checkdata = a1*0+1

    ## Write datafile
    # load config file and get parameters
    configFilename = sys.argv[1] #Load config file/parameters needed
    config = Config() # Create object and load file
    ok = config.loadFile( configFilename )
    if not ok:
        logger.info ('Error loading config file: "%s"' % configFilename)
        sys.exit(1)

    dataDir = config.getString('System', 'dataDir')
    floatID = os.uname()[1]
    now=datetime.utcnow() - timedelta(minutes=5) # so that raw and processed timestamps match
    telem_file = dataDir + floatID+'_TXimu_'+"{:%d%b%Y_%H%M%SUTC.dat}".format(now)
    with open(telem_file, 'w', newline='\n') as fp:
        fp.write('Hs,Tp,Dp,E,f,a1,b1,a2,b2,checkdata\n')
        fp.write(str(Hs)+'\n')
        fp.write(str(Tp)+'\n')
        fp.write(str(Dp)+'\n')
        fp.write(','.join(E.astype(str))+'\n')
        fp.write(','.join(f.astype(str))+'\n')
        fp.write(','.join(a1.astype(str))+'\n')
        fp.write(','.join(b1.astype(str))+'\n')
        fp.write(','.join(a2.astype(str))+'\n')
        fp.write(','.join(b2.astype(str))+'\n')
        fp.write(','.join(checkdata.astype(str))+'\n')

    logger.info('Data processing complete')


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

        # Start IMU processing
        logger.info('Start processing')
        x1 = threading.Thread(target=process_imu_thread, args=(fname,logger,imuFreq,), daemon=True)
        x1.start()

        time.sleep(0.1)
