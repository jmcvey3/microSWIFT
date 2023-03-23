import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
import matplotlib.dates as mpldt
plt.rcParams.update({'font.size': 14})
plt.close('all')

import dolfyn
from mhkit import wave


def read_gps(file):
    # Yost gets the XYZ coordinate system wrong: +Y and +Z are switched
    df = pd.read_csv(file, header=None, parse_dates=True, infer_datetime_format=True)
    
    ds = xr.Dataset(
        data_vars={'vel': (['dir','time'],
                             np.array([df[1], df[2]]),
                             {'units':'m s-1',
                              'long_name': 'Velocity'}),
                   'z': (['time'],
                             np.array(df[3]),
                             {'units':'m',
                              'long_name': 'Altitude',
                              'standard_name': 'altitude'}),
                   'lat': (['time'],
                           np.array(df[4]),
                           {'units':'degree_N',
                            'long_name': 'Latitude',
                            'standard_name': 'latitude'}),
                   'lon': (['time'],
                           np.array(df[5]),
                           {'units':'degree_E',
                            'long_name': 'Longitude',
                            'standard_name': 'longitude'}),
                   },
        coords = {'dir': ('dir', ['u','v']),
                  'time': ('time', df[0].values.astype('datetime64[ns]')),
                  })
    return ds

def read_yost_imu(file):
    # Yost gets the XYZ coordinate system wrong: +Y and +Z are switched
    df = pd.read_csv(file, header=None, parse_dates=True, infer_datetime_format=True)
    
    ds = xr.Dataset(
        data_vars={'accel_raw': (['dir','time'],
                             np.array([df[1], df[3], df[2]]),
                             {'units':'m s-2',
                              'long_name': 'acceleration vector'}),
                   'angrt': (['dir','time'],
                             np.array([df[4], df[6], df[5]]),
                             {'units':'rad s-1',
                              'long_name': 'angular velocity vector'}),
                   'mag': (['dir','time'],
                           np.array([df[7], df[9], df[8]]),
                           {'units':'gauss',
                            'long_name': 'magnetic field vector'}),
                   'quaternions': (['q','time'],
                                   np.array([df[10], df[12], df[11], df[13]]),
                                   {'units':'1',
                                    'long_name': 'quaternion vector'}),
                   'check_factor': (['time'],
                                    np.array(df[14]),
                                    {'units':'1',
                                     'long_name': 'check factor'}),
                   },
        coords = {'dir': ('dir', ['x','y','z']),
                  'q': ('q', [1, 2, 3, 4]),
                  'time': ('time', df[0].values.astype('datetime64[ns]')),
                  })
    return ds

def transform_data(ds):
    # Rotate the acclerations from the computed Quaterions
    r = R.from_quat(ds.quaternions.T)
    rpy = r.as_euler('XYZ', degrees=True)
    ds['accel'] = ds['accel_raw'].copy(deep=True)
    ds['accel'].data = r.apply(ds.accel.T).T

    ds['roll'] = xr.DataArray(rpy[:,0], 
                              dims=['time'], 
                              coords={'time':ds.time},
                              attrs={'units': 'degrees',
                                     'long_name': 'roll',
                                     'standard_name': 'platform_roll'})
    ds['pitch'] = xr.DataArray(rpy[:,1],
                              dims=['time'], 
                              coords={'time':ds.time},
                              attrs={'units': 'degrees',
                                     'long_name': 'pitch',
                                     'standard_name': 'platform_pitch'})
    ds['heading'] = xr.DataArray(rpy[:,0],
                              dims=['time'], 
                              coords={'time':ds.time},
                              attrs={'units': 'degrees',
                                     'long_name': 'heading',
                                     'standard_name': 'platform_orientation'})
    plt.figure()
    plt.plot(ds.time, ds['roll'], label='roll')
    plt.plot(ds.time, ds['pitch'], label='pitch')
    plt.legend()

    return ds

def RCfilter(b, fc=0.05, fs=2):
    """
    authors: @EJRainville, @AlexdeKlerk, @Viviana Castillo
    #TODO: fix docstr
    Helper function to perform RC filtering
    Input:
        - b, array of values to be filtered
        - fc, cutoff frequency, fc = 1/(2πRC)
        - fs, sampling frequency
    Output:
        - a, array of filtered input values
    """
    RC = (2*np.pi*fc)**(-1)
    alpha = RC / (RC + 1./fs)
    a = b.copy()
    for ui in np.arange(1,len(b)): # speed this up
        a[ui] = alpha * a[ui-1] + alpha * ( b[ui] - b[ui-1] )
    return a

def butter_lowpass_filter(data, fc, fs, order):
    normal_cutoff = fc / (0.5 * fs)
    # Get the filter coefficients 
    b, a = ss.butter(order, normal_cutoff, btype='low', analog=False)
    y = ss.filtfilt(b, a, data)
    return y

def filter_accel(ds):
    filt_freq = 0.0455  # max 20 second waves

    plt.figure()
    plt.plot(ds.time, ds['accel'][2] - ds['accel'][2].mean())
    plt.ylabel('Vertical [m/s]')
    plt.xlabel('Time')

    # Remove low frequency drift
    filt_factor = 3 # 5/3 # should be 5/3 for a butterworth filter
    if filt_freq == 0:
        hp = ds['accel'] - ds['accel'].mean()
    else:
        acclow = ds.accel.copy()
        acclow = butter_lowpass_filter(acclow, filt_factor*filt_freq, ds.fs, 2)
        hp = ds['accel'].data - acclow

    ds['accel'].data = hp
    plt.plot(ds.time, ds['accel'][2])

    # Integrate
    hp = ds['accel'].data
    dat = np.concatenate((np.zeros(list(hp.shape[:-1]) + [1]),
                      cumtrapz(hp, dx=1 / ds.fs, axis=-1)), axis=-1)
    # Run RC differentiator (high pass)
    dat = RCfilter(dat, filt_freq, ds.fs)

    ds['velacc'] = ds['accel'].copy() * 0
    ds['velacc'].data = dat
    ds['velacc'].attrs['units'] = 'm/s'
    ds['velacc'].attrs['long name'] = 'velocity vector from accelerometer'

    plt.figure()
    plt.plot(ds['velacc'][2])
    plt.ylabel('Vertical [m/s]')
    plt.xlabel('Time')

    return ds


def process_data(ds_imu, ds_gps, nbin, fs):
    ds_gps = ds_gps.interp(time=ds_imu.time)

    ## Wave height and period
    fft_tool = dolfyn.adv.api.ADVBinner(nbin, fs, n_fft=nbin/3, n_fft_coh=nbin/3)
    Sww = fft_tool.calc_psd(ds_imu['velacc'][2], freq_units='Hz')
    Sww = Sww.sel(freq=slice(0.0455, 1))
    Szz = Sww / (2*np.pi*Sww['freq'])**2
    pd_Szz = Szz.T.to_pandas()

    Hs = wave.resource.significant_wave_height(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)

    ## Wave direction and spread
    uva = xr.DataArray(np.array((ds_gps['vel'][0].data, ds_gps['vel'][1].data, ds_imu['accel'][2].data)),
                       dims=['dirUVA','time'],
                       coords={'dirUVA': ['u','v','accel'],
                               'time': ds_imu.time})
    psd = fft_tool.calc_psd(uva, freq_units='Hz')
    psd = psd.sel(freq=slice(0.0455, 1))
    Suu = psd.sel(S='Sxx')
    Svv = psd.sel(S='Syy')
    Saa = psd.sel(S='Szz')

    csd = fft_tool.calc_csd(uva, freq_units='Hz')
    csd = csd.sel(coh_freq=slice(0.0455, 1))
    Cua = csd.sel(C='Cxz').real
    Cva = csd.sel(C='Cyz').real

    a = Cua.values / np.sqrt((Suu+Svv)*Saa).values
    b = Cva.values / np.sqrt((Suu+Svv)*Saa).values
    theta = np.arctan(b/a)
    phi = np.sqrt(2*(1 - np.sqrt(a**2 + b**2)))
    phi = np.nan_to_num(phi) # fill missing data

    t = dolfyn.time.dt642date(Szz.time)
    direction = np.arange(len(t))
    spread = np.arange(len(t))
    for i in range(len(t)):
        direction[i] = 90 - np.rad2deg(np.trapz(theta[i], psd.freq)) # degrees CW from North
        spread[i] = np.rad2deg(np.trapz(phi[i], psd.freq))

    plt.figure()
    plt.loglog(Szz.freq, pd_Szz.mean(axis=1), label='vertical')
    m = -4
    x = np.logspace(-1, 0.5)
    y = 10**(-5)*x**m
    plt.loglog(x, y, '--', c='black', label='f^-4')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Energy Density [m^2/Hz]')
    plt.ylim((0.00001, 10))
    plt.legend()

    fig, ax = plt.subplots(2, figsize=(15,10))
    #ax = plt.figure(figsize=(20,10)).add_axes([.14, .14, .8, .74])
    ax[0].scatter(t, Hs)
    ax[0].set_xlabel('Time')
    ax[0].xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax[0].set_ylabel('Significant Wave Height [m]')

    #ax = plt.figure(figsize=(20,10)).add_axes([.14, .14, .8, .74])
    ax[1].scatter(t, Te)
    ax[1].set_xlabel('Time')
    ax[1].xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax[1].set_ylabel('Energy Period [s]')

    ax = plt.figure(figsize=(20,10)).add_axes([.14, .14, .8, .74])
    ax.scatter(t, direction, label='Wave direction (towards)')
    ax.scatter(t, spread, label='Wave spread')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax.set_ylabel('deg')
    plt.legend()

    ds_avg = xr.Dataset()
    ds_avg['Szz'] = Szz
    ds_avg['Suu'] = Suu
    ds_avg['Svv'] = Svv
    ds_avg['Saa'] = Saa
    ds_avg['Hs'] = Hs.to_xarray()['Hm0']
    ds_avg['Te'] = Te.to_xarray()['Te']
    ds_avg['Cua'] = Cua
    ds_avg['Cva'] = Cva
    ds_avg['a1'] = xr.DataArray(a, dims=['time', 'freq'])
    ds_avg['b1'] = xr.DataArray(b, dims=['time', 'freq'])
    ds_avg['direction'] = xr.DataArray(direction, dims=['time'])
    ds_avg['spread'] = xr.DataArray(spread, dims=['time'])

    return ds_avg


if __name__=='__main__':
    # Buoy deployment config
    fs = 4 # Hz
    nbin = int(fs*600) # 10 minute FFTs

    # Fetch IMU data
    files = glob(os.path.join('rPi','*_IMU*.dat'))
    ds = read_yost_imu(files[0])
    for i in range(len(files)):
        ds = xr.merge((ds, read_yost_imu(files[i])))
    ds.attrs['fs'] = fs
    ds = transform_data(ds)
    ds_imu = filter_accel(ds)
    ds_imu.to_netcdf('rPi_imu.raw.nc')

    # Fetch GPS data
    files = glob(os.path.join('rPi','*_IMU*.dat'))
    ds = read_gps(files[0])
    for i in range(len(files)):
        ds = xr.merge((ds, read_gps(files[i])))
    ds.attrs['fs'] = fs
    ds_gps = ds
    ds_gps.to_netcdf('rPi_gps.raw.nc')

    ds_waves = process_data(ds_imu, ds_gps, nbin, fs)
    ds_waves.to_netcdf('rPi.10m.nc')
