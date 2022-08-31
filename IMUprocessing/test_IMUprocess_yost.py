#--------------------------------------------------------------------------
#
# test_processIMUC.py
# reads in matlab data to use as input to processIMUC.c
#
#--------------------------------------------------------------------------
import sys
import os.path
import numpy as np
import array
import struct
import csv

import processIMU_lib


filename = 'raspberrypi_IMU_31Aug2022_202700UTC.dat'
with open(filename, 'r') as fp:
    data = list(csv.reader(fp, delimiter=','))
IMUdata = np.array(data)[:,1:-5].astype(float) # grab all data except time, quaternions and check

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
fs = np.double(4.) # sampling frequency

nv=np.size(axs)

print('IMU inputs:')
print(nv, axs, ays, azs, gxs, gys, gzs, mxs, mys, mzs, mxo, myo, mzo, Wd, fs)

# call processIMU
IMU_results = processIMU_lib.main_processIMU(nv, axs, ays, azs, gxs, gys, gzs, 
                                                 mxs, mys, mzs, mxo, myo, mzo, Wd, fs)
Sigwave_Height = IMU_results[0]
Peakwave_Period = IMU_results[1]
Peakwave_dirT = IMU_results[2]
WaveSpectra_Energy = np.squeeze(IMU_results[3])
WaveSpectra_Freq   = np.squeeze(IMU_results[4])
WaveSpectra_a1 = np.squeeze(IMU_results[5])
WaveSpectra_b1 = np.squeeze(IMU_results[6])
WaveSpectra_a2 = np.squeeze(IMU_results[7])
WaveSpectra_b2 = np.squeeze(IMU_results[8])
checkdata = WaveSpectra_a1*0+1
print('IMU results:')
print('Hs=',Sigwave_Height,'Tp=',Peakwave_Period,'Dp=',Peakwave_dirT)
print('E',WaveSpectra_Energy)
print('f',WaveSpectra_Freq)
print('a1',WaveSpectra_a1)
print('b1',WaveSpectra_b1)
print('a2',WaveSpectra_a2)
print('b2',WaveSpectra_b2)
print('checkdata',checkdata)
