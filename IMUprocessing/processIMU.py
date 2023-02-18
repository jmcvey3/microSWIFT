import numpy as np
import csv
import processIMU_lib


filename = "../data/raspberrypi_IMU_19Mar2022_042500UTC.dat"
with open(filename, 'r') as f:
    IMUdata = np.array(csv.reader(f, delimiter=','))

axs = IMUdata[:,0]
ays = IMUdata[:,1]
azs = IMUdata[:,2]
gxs = IMUdata[:,3]
gys = IMUdata[:,4]
gzs = IMUdata[:,5]
mxs = IMUdata[:,6]
mys = IMUdata[:,7]
mzs = IMUdata[:,8]

mxo = np.double(60.) 
myo = np.double(60.) 
mzo = np.double(120.)  
Wd = np.double(0.) 
fs = np.double(4.) 

nv=np.size(axs)

print('IMU inputs:')
print(nv, axs, ays, azs, gxs, gys, gzs,mxs, mys, mzs, mxo, myo, mzo, Wd, fs)

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
