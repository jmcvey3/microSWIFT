#! /usr/bin/python3

#imports
import serial, sys, os
import numpy as np
from struct import *
from logging import *
from datetime import datetime
import time as t
import RPi.GPIO as GPIO
import pynmea2
from time import sleep
import threading

#my imports
from config3 import Config
import process_data


def calc_checksum(message):
    checksum = 0
    for char in message:
        checksum ^= ord(char)
    checksum = hex(checksum)[2:]
    return '$' + message + '*' + str(checksum) + '\r\n'

#load config file and get parameters
configFilename = sys.argv[1] #Load config file/parameters needed
config = Config() # Create object and load file
ok = config.loadFile( configFilename )
if( not ok ):
    logger.info ('Error loading config file: "%s"' % configFilename)
    sys.exit(1)
    
#system parameters
dataDir = config.getString('System', 'dataDir')
floatID = os.uname()[1]
#floatID = config.getString('System', 'floatID') 
sensor_type = config.getInt('System', 'sensorType')
badValue = config.getInt('System', 'badValue')
numCoef = config.getInt('System', 'numCoef')
port = config.getInt('System', 'port')
payload_type = config.getInt('System', 'payloadType')
burst_seconds = config.getInt('System', 'burst_seconds')
burst_time = config.getInt('System', 'burst_time')
burst_int = config.getInt('System', 'burst_interval')
call_int = config.getInt('Iridium', 'call_interval')
call_time = config.getInt('Iridium', 'call_time')


#GPS parameters 
gps_port = config.getString('GPS', 'port')
start_baud = config.getInt('GPS', 'start_baud')
baud = config.getInt('GPS', 'baud')
gps_freq = config.getInt('GPS', 'GPS_frequency') #currently not used, hardcoded at 4 Hz (see init_gps function)
#numSamplesConst = config.getInt('System', 'numSamplesConst')
gps_samples = gps_freq*burst_seconds
#gpsGPIO = config.getInt('GPS', 'gpsGPIO')
gps_timeout = config.getInt('GPS','timeout')

##setup GPIO and initialize - GPS running entire time - jrm
#GPIO.setmode(GPIO.BCM)
#GPIO.setwarnings(False)
#GPIO.setup(modemGPIO,GPIO.OUT)
#GPIO.setup(gpsGPIO,GPIO.OUT)
#GPIO.output(gpsGPIO,GPIO.HIGH) #set GPS enable pin high to turn on and start acquiring signal

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def init_gps():
    nmea_time=''
    nmea_date=''
    #set GPS enable pin high to turn on and start acquiring signal
    #GPIO.output(gpsGPIO,GPIO.HIGH)
    try:
        logger.info('initializing GPS')
        logger.info("Trying GPS serial port at %s" % baud)
        ser=serial.Serial(gps_port,start_baud,timeout=1)
        logger.info("Connected")
        try:
            #set device baud rate
            baud_base_command = 'PMTK251,'+str(baud)
            baud_command = calc_checksum(baud_base_command)
            logger.info("Setting baud rate to %s: %s" % (baud, baud_command))
            ser.write(baud_command.encode())
            sleep(1)

            #switch ser port to baud rate
            ser.baudrate=baud
            logger.info("switching to %s on port %s" % (baud, gps_port))

            ## Set output sentence to GPGGA and GPVTG, plus GPRMC once every 4 positions (See GlobalTop PMTK command packet PDF)
            output_base_command = 'PMTK314,0,4,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0'
            output_command = calc_checksum(output_base_command)
            logger.info('setting NMEA output sentence %s' % (output_command))
            ser.write(output_command.encode())
            sleep(1)
        
            ## Set sampling frequency
            fs = str(int(1/gps_freq*1000)) # sampling period in milliseconds
            fs_base_command = 'PMTK220,'+fs
            fs_command = calc_checksum(fs_base_command)
            logger.info("setting GPS to %s Hz rate: %s" % (gps_freq, fs_command))
            ser.write(fs_command.encode())
            sleep(1)
        except Exception as e:
            logger.info(e)
            return ser, False, nmea_time, nmea_date

    except Exception as e:
        logger.info(e)
        return ser, False, nmea_time, nmea_date

    #read lines from GPS serial port and wait for fix
    try:
        #loop until timeout dictated by gps_timeout value (seconds)
        timeout=t.time() + gps_timeout
        while t.time() < timeout:
            ser.flush()
            ser.read_until('\n'.encode())
            newline=ser.readline().decode('utf-8')
            logger.info(newline)
            if not 'GPGGA' in newline:
                newline=ser.readline().decode('utf-8')
                if 'GPGGA' in newline:
                    logger.info('found GPGGA sentence')
                    logger.info(newline)
                    gpgga=pynmea2.parse(newline,check=True)
                    logger.info('GPS quality= %d' % gpgga.gps_qual)
                    #check gps_qual value from GPGGS sentence. 0=invalid,1=GPS fix,2=DGPS fix
                    if gpgga.gps_qual > 0:
                        logger.info('GPS fix acquired')
                        # get date and time from GPRMC sentence - GPRMC reported only once every second
                        # 8 lines at 4 Hz (GGA + VTG)
                        for i in range(2*gps_freq):
                            newline=ser.readline().decode('utf-8')
                            if 'GPRMC' in newline:
                                logger.info('found GPRMC sentence')

                                try:
                                    gprmc=pynmea2.parse(newline)
                                    nmea_time=gprmc.timestamp
                                    nmea_date=gprmc.datestamp
                                    logger.info("nmea time: %s" %nmea_time)
                                    logger.info("nmea date: %s" %nmea_date)
                                    
                                    #set system time
                                    try:
                                        logger.info("setting system time from GPS: %s %s" %(nmea_date, nmea_time))
                                        os.system('sudo timedatectl set-timezone UTC')
                                        os.system('sudo date -s "%s %s"' %(nmea_date, nmea_time))
                                        #os.system('sudo hwclock -w --verbose') # external RTC
                        				
                                        logger.info("GPS initialized")
                                        return ser, True, nmea_time, nmea_date

                                    except Exception as e:
                                        logger.info(e)
                                        logger.info('error setting system time')
                                        continue	
                                except Exception as e:
                                    logger.info(e)
                                    logger.info('error parsing nmea sentence')
                                    continue
                        # return False if gps fix but time not set	
                        return ser, False, nmea_time, nmea_date
            sleep(1)
        #return False if loop is allowed to timeout
        return ser, False, nmea_time, nmea_date
    except Exception as e:
        logger.info("Error setting up GPS")
        logger.info(e)
        return ser, False, nmea_time, nmea_date

#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def record_gps(ser,fname):

    #initialize empty numpy array and fill with bad values
    ts = []
    u = np.empty(gps_samples)
    u.fill(badValue)
    v = np.empty(gps_samples)
    v.fill(badValue)
    z = np.empty(gps_samples)
    z.fill(badValue)
    lat = np.empty(gps_samples)
    lat.fill(badValue)
    lon = np.empty(gps_samples)
    lon.fill(badValue)
    
    try:
        ser.flush()
        with open(fname, 'w',newline='\n') as gps_out:
            
            logger.info('open file for writing: %s' %fname)
            t_end = t.time() + burst_seconds #get end time for burst
            ipos = 0
            ivel = 0
            qc = False #assume data is good
            while t.time() <= t_end or ipos < gps_samples or ivel < gps_samples:
                newline=ser.readline().decode()
                gps_out.write(newline)
        
                if "GPGGA" in newline:
                    gpgga = pynmea2.parse(newline,check=True)   #grab gpgga sentence and parse
                    #check to see if we have lost GPS fix, and if so, continue to loop start. a badValue will remain at this index
                    qc = gpgga.gps_qual < 1
                    if qc:
                        logger.info('lost GPS fix, sample not recorded. Waiting 10 seconds')
                        sleep(10)
                        ipos+=1
                        continue
                    ts.append(gpgga.timestamp.strftime("%H:%M:%S.%f"))
                    z[ipos] = gpgga.altitude #units are meters
                    lat[ipos] = gpgga.latitude
                    lon[ipos] = gpgga.longitude
                    ipos+=1
                elif "GPVTG" in newline:
                    if qc:
                        continue
                    gpvtg = pynmea2.parse(newline,check=True)   #grab gpvtg sentence
                    if gpvtg.true_track:
                        u[ivel] = 0.2778 * gpvtg.spd_over_grnd_kmph*np.cos(gpvtg.true_track) #get u component of SOG and convert to m/s
                        v[ivel] = 0.2778 * gpvtg.spd_over_grnd_kmph*np.sin(gpvtg.true_track) #get v component of SOG and convert to m/s
                        ivel+=1
                else: #if not GPGGA or GPVTG, continue to start of loop
                    continue
            
                #if burst has ended but we are close to getting the right number of samples, continue for a short while
                if t.time() >= t_end and 0 < gps_samples-ipos and gps_samples-ipos <= 10:
                    continue
                elif ipos == gps_samples and ivel == gps_samples:
                    break
                
        badpts = len(np.where(z == 999)) #index of bad values if lost GPS fix. Should be same for u and v
    
        logger.info('number of GPGGA samples = %s' %ipos)
        logger.info('number of GPVTG samples = %s' %ivel)
        logger.info('number of bad samples %d' %badpts)
                        
        return u,v,z,lat,lon,ts
        
    except Exception as e:
        logger.info(e, exc_info=True)
        return u,v,z,lat,lon,ts


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
def process_gps_thread(fname,u,v,z,lat,lon,ts):
    # overwrite gps nmea file with data
    with open(fname,'w', newline='\n') as fp:
        for i in range(len(u)):
            fp.write('%s,%f,%f,%f,%f,%f\n' %(ts[i],u[i],v[i],z[i],lat[i],lon[i]))

    process_data.main(u,v,z,lat,lon)


#------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------
#set up logging, initialize GPS, and record data unless importing as a module
if __name__ == "__main__":
    
    #set up logging
    logDir = config.getString('Loggers', 'logDir')
    LOG_LEVEL = config.getString('Loggers', 'DefaultLogLevel')
    #format log messages (example: 2020-11-23 14:31:00,578, recordGPS - info - this is a log message)
    #NOTE: TIME IS SYSTEM TIME
    LOG_FORMAT = ('%(asctime)s, %(filename)s - [%(levelname)s] - %(message)s')
    #log file name (example: home/pi/microSWIFT/recordGPS_23Nov2020.log)
    LOG_FILE = (logDir + '/' + 'recordGPS' + '_' + datetime.strftime(datetime.now(), '%d%b%Y') + '.log')
    logger = getLogger('system_logger')
    logger.setLevel(LOG_LEVEL)
    logFileHandler = FileHandler(LOG_FILE)
    logFileHandler.setLevel(LOG_LEVEL)
    logFileHandler.setFormatter(Formatter(LOG_FORMAT))
    logger.addHandler(logFileHandler)
    
    logger.info("---------------recordGPS.py------------------")
    logger.info('python version {}'.format(sys.version))
    
    logger.info('microSWIFT configuration:')
    logger.info('float ID: {0}, payload type: {1}, sensors type: {2}, '.format(floatID, payload_type, sensor_type))
    logger.info('burst seconds: {0}, burst interval: {1}, burst time: {2}'.format(burst_seconds, burst_int, burst_time))
    logger.info('gps sample rate: {0}, call interval {1}, call time: {2}'.format(gps_freq, call_int, call_time))
    #call function to initialize GPS
    ser, gps_initialized, time, date = init_gps()
    
    if gps_initialized:
        logger.info('waiting for burst start')
        while True:
            #burst start conditions
            now=datetime.utcnow()
            if now.minute % burst_int == 0 and now.second == 0:
                
                logger.info("starting burst")
                #create file name
                fname = dataDir + floatID + '_GPS_'+"{:%d%b%Y_%H%M%SUTC.dat}".format(datetime.utcnow())
                logger.info("file name: %s" %fname)
                #call record_gps	
                u,v,z,lat,lon,ts = record_gps(ser,fname)
                
                try:
                    if os.path.isfile(fname) and os.path.getsize(fname) > 0:
                        #call data processing script
                        logger.info('starting to process data')
                        #print(u.shape)
                        x_gps = threading.Thread(target=process_gps_thread, args=(fname,u,v,z,lat,lon,ts), daemon=True)
                        x_gps.start()
                    else:
                        logger.info('data file does not exist or does not contain enough data for processing')	
                except OSError as e:
                    logger.info(e)
                    sys.exit(1)

            else:
                sleep(0.25)
    else:
        logger.info("GPS not initialized, exiting")
        sys.exit(1)
