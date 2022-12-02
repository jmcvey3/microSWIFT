"""
The main operational script that runs on the microSWIFT V1 wave buoys.

This script sequences the microSWIFT data collection, post-processing,
and telemetering. Its core task is to schedule these events, ensuring
that the buoy is in the appropriate record or send window based on the
user-defined settings. The process flow is summarized as follows:

    1. Record GPS and record IMU concurrently; write to .dat files
    2. Read the raw data into memory and process it into a wave solution
    3. Create a payload and pack it into an .sbd message; telemeter the
       message to the SWIFT server.

Author(s):
EJ Rainville (UW-APL), Alex de Klerk (UW-APL), Jim Thomson (UW-APL),
Viviana Castillo (UW-APL), Jacob Davis (UW-APL)

microSWIFT is licensed under the GNU General Public License v3.0.

"""

import concurrent.futures
import os
import sys
import time
from datetime import datetime

import numpy as np

from .accoutrements import imu
from .accoutrements import gps
from .accoutrements import sbd
from .accoutrements import telemetry_stack
from .processing.gps_waves import gps_waves
from .processing.uvza_waves import uvza_waves
from .processing.collate_imu_and_gps import collate_imu_and_gps
from .utils import config
from .utils import log
from .utils import utils


####TODO: Update this block when EJ finishes the config integration ####
#Define Config file name and load file
CONFIG_FILENAME = r'/home/pi/microSWIFT/utils/Config.dat'
config = config.Config() # Create object and load file
loaded = config.loadFile(CONFIG_FILENAME)
if not loaded:
    print("Error loading config file")
    sys.exit(1)


# System Parameters
DATA_DIR = config.getString('System', 'dataDir')
FLOAT_ID = os.uname()[1]
SENSOR_TYPE = config.getInt('System', 'sensorType')
BAD_VALUE = config.getInt('System', 'badValue')
NUM_COEF = config.getInt('System', 'numCoef')
PORT = config.getInt('System', 'port')
PAYLOAD_TYPE = config.getInt('System', 'payloadType')
BURST_SECONDS = config.getInt('System', 'burst_seconds')
BURST_TIME = config.getInt('System', 'burst_time')
BURST_INT = config.getInt('System', 'burst_interval')


# GPS parameters
GPS_FS = config.getInt('GPS', 'gps_frequency') #TODO: currently not used, hardcoded at 4 Hz (see init_gps function)
# IMU parameters
IMU_FS = config.getFloat('IMU', 'imuFreq') #TODO: NOTE this has been changed to 12 from 12.5 (actual) to obtain proper # of pts in processing

#Compute number of bursts per hour
NUM_BURSTS = int(60 / BURST_INT)


#Generate lists of burst start and end times based on parameters from Config file
start_times = [BURST_TIME + i*BURST_INT for i in range(NUM_BURSTS)]
end_times = [start_times[i] + BURST_SECONDS/60 for i in range(NUM_BURSTS)]

#TODO: add these to the config
WAVE_PROCESSING_TYPE = 'gps_waves'
WAIT_LOG_MESSAGE_INTERVAL = 10
########################################################################

# Initialize the logger to keep track of running tasks. These will print
# directly to the microSWIFT's log file. Then log the configuration.
logger = log.init()

##############TODO: have the config file spit this out? ################
logger.info(log.header(''))
logger.info(('Booted up. microSWIFT configuration: \n'
             f'float ID: {FLOAT_ID}, payload type: {PAYLOAD_TYPE},'
             f' sensor type: {SENSOR_TYPE}, burst seconds: {BURST_SECONDS},'
             f' burst interval:  {BURST_INT}, burst time: {BURST_TIME}'))
########################################################################


# Define loop and wait counters: `loop_count` keeps track of the number
# of duty cycles and `wait_count` iterates the wait log message.
loop_count = 1
wait_count = 0

# Initialize the telemetry stack if it does not exist yet. This is a
# text file that keeps track of the SBD message filenames that remain to
# be sent to the SWIFT server. If messages are not sent during a send
# window, the message filename will be pushed onto the stack and the
# script will attempt to send them during the next window. This ensures
# recent messages will be sent first.
telemetry_stack.init()
logger.info(f'Number of messages in queue: {telemetry_stack.get_length()}')


while True:
    current_min = datetime.utcnow().minute + datetime.utcnow().second/60
    duty_cycle_start_time = datetime.now()
    recording_complete = False

    # If the current time is within any record window (between start
    # and end time) record the imu and gps data until the end of the
    # window. These tasks are submitted concurrently.
    for i in np.arange(len(start_times)):
        if start_times[i] <= current_min < end_times[i]:

            logger.info(log.header(f'Iteration {loop_count}'))

            end_time = end_times[i]

            # Define next start time to enter into the send window.
            current_start = datetime.utcnow().replace(minute=start_times[i],
                                                      second = 0,
                                                      microsecond=0)
            next_start = current_start + datetime.timedelta(minutes=BURST_INT)

            # Run record_gps.py and record_imu.py concurrently with
            # asynchronous futures. This is a two-step call that
            # requires scheduling the tasks then returning the result
            # from each Future instance. Flip the`recording_complete`
            # state when the tasks are completed.
            with concurrent.futures.ThreadPoolExecutor() as executor:
                record_gps_future = executor.submit(gps.record, end_times[i])
                record_imu_future = executor.submit(imu.record, end_times[i])

                gps_file, gps_initialized = record_gps_future.result()
                imu_file, imu_initialized = record_imu_future.result()

            recording_complete = True
            break

    # Process the data into a wave estimate based on the specified
    # processing type. Check that the appropriate modules are
    # initialized, otherwise log a warning and fill the all of the
    # results with bad values of the expected length.
    if recording_complete is True:
        logger.info('Starting Processing')
        begin_processing_time = datetime.now()

        # GPS waves processing: convert the raw GPS data to East-West
        # velocity (u), North-South velocity (v), and elevation (z),
        # then produce a wave estimate.
        if WAVE_PROCESSING_TYPE == 'gps_waves' and gps_initialized:
            gps_vars = gps.to_uvz(gps_file)
            Hs, Tp, Dp, E, f, a1, b1, a2, b2, check \
                                                    = gps_waves(gps_vars['u'],
                                                                gps_vars['v'],
                                                                gps_vars['z'],
                                                                GPS_FS)
            logger.info('gps_waves.py executed')

            #TODO: This solution is not great but can be sorted out later:
            u = gps_vars['u']
            v = gps_vars['v']
            z = gps_vars['z']
            lat = gps_vars['lat']
            lon = gps_vars['lon']

        # UVZA waves processing: convert the raw GPS data to East-West
        # velocity (u), North-South velocity (v), and elevation (z);
        # integrate the raw IMU to xyz displacements in the body or
        # earth frame; then collate these variables onto the same time
        # array and produce a wave estimate. The first two-minutes
        # are zeroed-out to remove filter ringing.
        elif WAVE_PROCESSING_TYPE == 'uvza_waves' and gps_initialized \
                                                        and imu_initialized:
            gps_vars = gps.to_uvz(gps_file)
            imu_vars =imu.to_xyz(imu_file, IMU_FS)
            imu_collated, gps_collated = collate_imu_and_gps(imu_vars, gps_vars)

            ZERO_POINTS = int(np.round(120*IMU_FS))
            Hs, Tp, Dp, E, f, a1, b1, a2, b2, check  \
                                = uvza_waves(gps_collated['u'][ZERO_POINTS:],
                                             gps_collated['v'][ZERO_POINTS:],
                                             imu_collated['pz'][ZERO_POINTS:],
                                             imu_collated['az'][ZERO_POINTS:],
                                             IMU_FS)
            logger.info('uvza_waves.py executed.')

            # TODO: This solution is not great but can be sorted out later:
            u = gps_vars['u']
            v = gps_vars['v']
            z = gps_vars['z']
            lat = gps_vars['lat']
            lon = gps_vars['lon']

        else:
            logger.info(('A wave solution cannot be created; either the'
                f' specified processing type (={WAVE_PROCESSING_TYPE}) is'
                f' invalid, or either or both of the sensors failed to'
                f' initialize (GPS initialized={gps_initialized}, IMU'
                f' initialized={imu_initialized}). Entering bad values for'
                f' the wave products (={BAD_VALUE}).'))
            u, v, z, lat, lon, Hs, Tp, Dp, E, f, a1, b1, a2, b2, check \
                                = utils.fill_bad_values(badVal=BAD_VALUE,
                                                        spectralLen=NUM_COEF)

        # check lengths of spectral quanities:
        if len(E)!=NUM_COEF or len(f)!=NUM_COEF:
            logger.info(('WARNING: the length of E or f does not match the'
                         f' specified number of coefficients, {NUM_COEF};'
                         f' (len(E)={len(E)}, len(f)={len(f)})'))

        # Compute the mean of the GPS output. The last reported
        # position is sent to the server.
        u_mean = np.nanmean(u)
        v_mean = np.nanmean(v)
        z_mean = np.nanmean(z)
        last_lat = utils.get_last(BAD_VALUE, lat)
        last_lon = utils.get_last(BAD_VALUE, lon)

        # Populate the voltage, temperature, salinity fields with place-
        # holders. These modules will be incorporated in the future.
        voltage = 0
        temperature = 0.0
        salinity = 0.0

        logger.info(('Processing section took {}'.format(datetime.now()
                     - begin_processing_time)))


        # Pack the payload data into a short burst data (SBD) message
        # to be telemetered to the SWIFT server. The SBD filenames are
        # entered into a stack (last in, first out) in the order in
        # which they were created such that the most recent messages
        # are sent first.
        logger.info('Creating TX file and packing payload data')
        tx_filename, payload_data \
                    = sbd.createTX(Hs, Tp, Dp, E, f, a1, b1, a2, b2, check,
                                   u_mean, v_mean, z_mean, last_lat, last_lon,
                                   temperature, salinity, voltage)

        # Push the newest SBD filenames onto the stack and return the
        # updated list of payload filenames. The list must be flipped
        # to be consistent with the LIFO ordering. Iterate through the
        # stack and send until the current time window is up. Update the
        # stack each loop (if a send is successful) and re-write the
        # payload filenames to the stack file.
        payload_filenames = telemetry_stack.push(tx_filename)
        payload_filenames_LIFO = list(np.flip(payload_filenames))

        logger.info(f'Number of Messages to send: {len(payload_filenames)}')

        messages_sent = 0
        for tx_file in payload_filenames_LIFO:
            if datetime.utcnow() < next_start:
                logger.info(f'Opening TX file from payload list: {tx_file}')

                with open(tx_file, mode='rb') as file:
                    payload_data = file.read()

                successful_send = sbd.send(payload_data, next_start)

                if successful_send is True:
                    del payload_filenames[-1]
                    messages_sent += 1
            else:
                # Exit if the send window has expired.
                break

        telemetry_stack.write(payload_filenames)


        # End of the loop; log the send statistics and increment the
        # counters for the next iteration.
        messages_remaining = len(payload_filenames) - messages_sent
        logger.info((f'Messages Sent: {int(messages_sent)}; '
                     f'Messages Remaining: {int(messages_remaining)}'))
        logger.info(('microSWIFT.py took {}'.format(datetime.now() 
                     - begin_processing_time)))

        loop_count += 1
        wait_count = 0


    # The current time is not within the defined record window. Skip
    # telemetry and sleep until a window is entered. Log this
    # information at the specified interval (in seconds).
    else:
        time.sleep(1)
        wait_count += 1
        if wait_count % WAIT_LOG_MESSAGE_INTERVAL == 0:
            logger.info('Waiting to enter record window')
        continue
