import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R

import dolfyn
from mhkit import wave


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
    return ds


def RCfilter(b, fc=0.05, fs=2):
    """
    authors: @EJRainville, @AlexdeKlerk, @Viviana Castillo

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


def filter_accel(ds, filt_freq):
    # Remove low frequency drift
    filt_factor = 5/3 # should be 5/3 for a butterworth filter
    if filt_freq == 0:
        hp = ds['accel'] - ds['accel'].mean()
    else:
        acclow = ds.accel.copy()
        acclow = butter_lowpass_filter(acclow, filt_factor*filt_freq, ds.fs, 2)
        hp = ds['accel'].data - acclow

    ds['accel'].data = hp

    # Integrate
    dat = np.concatenate((np.zeros(list(hp.shape[:-1]) + [1]),
                      cumtrapz(hp, dx=1 / ds.fs, axis=-1)), axis=-1)
    # Run RC differentiator (high pass)
    dat = RCfilter(dat, filt_freq, ds.fs)

    ds['velacc'] = ds['accel'].copy() * 0
    ds['velacc'].data = dat
    ds['velacc'].attrs['units'] = 'm/s'
    ds['velacc'].attrs['long name'] = 'velocity vector from accelerometer'

    return ds


if __name__=='__main__':
    # Buoy deployment config
    fs = 4 # Hz
    nbin = int(fs*600) # 10 minute FFTs
    filt_freq = 0.0455  # max 22 second waves

    files = glob(os.path.join('data','*_IMU*.dat'))
    ds = read_yost_imu(files[0])
    for i in range(len(files)):
        ds = xr.merge((ds, read_yost_imu(files[i])))

    ds.attrs['fs'] = fs
    ds = transform_data(ds)
    ds = filter_accel(ds, filt_freq)

    ## Using dolfyn to create spectra
    fft_tool = dolfyn.adv.api.ADVBinner(nbin, fs, n_fft=nbin, n_fft_coh=nbin)
    Sww = fft_tool.calc_psd(ds['velacc'][2], freq_units='Hz')
    Szz = Sww / (2*np.pi*Sww['freq'])**2
    Szz = Szz.sel(freq=slice(0.0455, None))
    pd_Szz = Szz.T.to_pandas()

    rpa = xr.DataArray(np.array((ds['roll'].data, ds['pitch'].data, ds['accel'][2].data)),
                    dims=['dirRPA','time'],
                    coords={'dirRPA': ['roll','pitch','accel'],
                            'time': ds.time})
    psd = fft_tool.calc_psd(rpa, freq_units='Hz')
    Sxx = psd.sel(S='Sxx')
    Syy = psd.sel(S='Syy')
    Saa = psd.sel(S='Szz')
    csd = fft_tool.calc_csd(rpa, freq_units='Hz')


    ## Wave analysis using MHKiT
    Hs = wave.resource.significant_wave_height(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)

    ## Wave direction and spread
    Cxz = csd.sel(C='Cxz').real
    Cyz = csd.sel(C='Cyz').real

    a = Cxz/np.sqrt((Sxx+Syy)*Saa)
    b = Cyz/np.sqrt((Sxx+Syy)*Saa)
    theta = np.arctan(b/a) * (180/np.pi) # degrees CCW from East
    #theta = dolfyn.tools.misc.convert_degrees(theta) # degrees CW from North
    phi = np.sqrt(2*(1 - np.sqrt(a**2 + b**2))) * (180/np.pi)
    phi = phi.fillna(0) # fill missing data

    direction = np.arange(len(Szz.time))
    spread = np.arange(len(Szz.time))
    for i in range(len(Szz.time)):
        direction[i] = np.trapz(theta[i], psd.freq)
        spread[i] = np.trapz(phi[i], psd.freq)

    ds_avg = xr.Dataset()
    ds_avg['Szz'] = Szz
    ds['rpa'] = rpa
    ds_avg['Srr'] = Sxx
    ds_avg['Spp'] = Syy
    ds_avg['Saa'] = Saa
    ds_avg['Hs'] = Hs.to_xarray()['Hm0']
    ds_avg['Te'] = Te.to_xarray()['Te']
    ds_avg['Cra'] = Cxz
    ds_avg['Cpa'] = Cyz
    ds_avg['a1'] = a
    ds_avg['b1'] = b
    ds_avg['direction'] = xr.DataArray(direction, dims=['time'])
    ds_avg['spread'] = xr.DataArray(spread, dims=['time'])

    ds.to_netcdf('data/rPi_imu.nc')
    ds_avg.to_netcdf('data/rPi_imu_spec.nc')
