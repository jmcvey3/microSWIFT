#! /usr/bin/python3

#standard imports 
import time, os, sys
from datetime import datetime
import serial
from logging import *

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
imuFreq=config.getFloat('IMU', 'imuFreq')
imu_samples = imuFreq*burst_seconds


#------------------------------------------------------------
#Yost iMU setup
# open serial port
ser = serial.Serial(port='/dev/ttyACM0',baudrate=115200)
# serial commands
ser.write(':80,0,45,65,66,67,255,255,255\r\n'.encode()) # sensor config
ser.write(':82,250000,-1,0\r\n'.encode()) # timing
ser.write(':85\r\n'.encode()) # start
#-----------------------------------------------------------------------------
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
    
    
    now=datetime.utcnow()
    if  now.minute == burst_time or now.minute % burst_interval == 0 and now.second == 0:
        

        #Get data
        # pull serial data (written in lines, not bytes)
        data0 = ser.readline()
        data45 = ser.readline()
        data65 = ser.readline()
        data66 = ser.readline()
        data67 = ser.readline()
        
        
        logger.info('starting burst')
        
        #create new file for new burst interval 
        fname = dataDir + floatID + '_IMU_'+'{:%d%b%Y_%H%M%SUTC.dat}'.format(datetime.utcnow())
        logger.info('file name: %s' %fname)
             
        with open(fname, 'w',newline='\n') as imu_out:
            logger.info('open file for writing: %s' %fname)
            t_end = time.time() + burst_seconds #get end time for burst
            isample=0
            while time.time() <= t_end or isample < imu_samples:
        
                try:
                    gyro_x, gyro_y, gyro_z = data65
                    accel_x, accel_y, accel_z = data66
                    mag_x, mag_y, mag_z = data67
                    
                except Exception as e:
                    logger.info(e)
                    logger.info('error reading IMU data')
         
                timestamp='{:%Y-%m-%d %H:%M:%S}'.format(datetime.utcnow())

                info = { 'gyro_raw':data65, 'accel_raw':data66, 'comp_raw':data67, \
                    'quat':data0,'conf_fac':data45, }

                imu_out.write('%s,%f\n' %(timestamp,info))
                imu_out.flush()
        
                isample = isample + 1
                
               
                if time.time() >= t_end and 0 < imu_samples-isample <= 40:
                    continue
                elif time.time() > t_end and imu_samples-isample > 40:
                    break
                
            
            logger.info('end burst')
            logger.info('IMU samples %s' %isample)  
               
            
#potentially ser.close()