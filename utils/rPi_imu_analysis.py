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

from mhkit import wave
import dolfyn


slc_freq = slice(0.0455, 1)

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


def calc_tilt(pitch, roll):
    tilt = np.rad2deg(np.arctan(
        np.sqrt(np.tan(np.deg2rad(roll)) ** 2 + np.tan(np.deg2rad(pitch)) ** 2)
    ))
    return tilt


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
    # Filter factor should be 5/3 for a perfectly centered or gimballed IMU
    # Tune filt_factor to buoy if IMU is not centered (Teng 2002)
    filt_factor = 5/3
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


def process_data(ds, nbin, fs):
    ## Using dolfyn to create spectra
    fft_tool = dolfyn.adv.api.ADVBinner(nbin, fs, n_fft=nbin, n_fft_coh=nbin)
    Sww = fft_tool.calc_psd(ds['velacc'][2], freq_units='Hz')
    Szz = Sww / (2*np.pi*Sww['freq'])**2
    Szz = Szz.sel(freq=slc_freq)
    pd_Szz = Szz.T.to_pandas()
    
    # Non-deterministic approach:
    # Saa = fft_tool.calc_psd(ds['accel'][2], freq_units='Hz')
    # Szz = Saa / (2*np.pi*Saa['freq'])**4
    # Szz = Szz.sel(freq=slc_freq)
    # pd_Szz = Szz.T.to_pandas()

    rpa = xr.DataArray(np.array((ds['roll'].data, ds['pitch'].data, ds['accel'][2].data)),
                    dims=['dirRPA','time'],
                    coords={'dirRPA': ['roll','pitch','accel'],
                            'time': ds.time})
    psd = fft_tool.calc_psd(rpa, freq_units='Hz').sel(freq=slc_freq)
    Sxx = psd.sel(S='Sxx')
    Syy = psd.sel(S='Syy')
    Saa = psd.sel(S='Szz')
    csd = fft_tool.calc_csd(rpa, freq_units='Hz').sel(coh_freq=slc_freq)


    ## Wave analysis using MHKiT
    Hs = wave.resource.significant_wave_height(pd_Szz)
    Tm = wave.resource.average_wave_period(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)
    Tp = wave.resource.peak_period(pd_Szz)

    ## Wave direction and spread
    # Must make sure "coh_freq" == "freq"
    Cxz = csd.sel(C='Cxz').real
    Cyz = csd.sel(C='Cyz').real

    a = Cxz.values / np.sqrt((Sxx+Syy)*Saa)
    b = Cyz.values / np.sqrt((Sxx+Syy)*Saa)
    theta = np.arctan(b/a) # degrees CCW from East, "to" convention
    phi = np.sqrt(2*(1 - np.sqrt(a**2 + b**2)))
    phi = np.nan_to_num(phi) # fill missing data

    direction = np.arange(len(Szz.time))
    spread = np.arange(len(Szz.time))
    for i in range(len(Szz.time)):
        # degrees CW from North, "to" convention (90 - X)
        # degrees CW from North, "from" convention (-90 - X)
        direction[i] = -90 - np.rad2deg(np.trapz(theta[i], psd.freq))
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
    plt.savefig('fig/wave_spectrum.imu.png')

    t = dolfyn.time.dt642date(Szz.time)
    fig, ax = plt.subplots(2, figsize=(10,7))
    ax[0].scatter(t, Hs)
    ax[0].set_xlabel('Time')
    ax[0].xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax[0].set_ylabel('Significant Wave Height [m]')

    ax[1].scatter(t, Te)
    ax[1].set_xlabel('Time')
    ax[1].xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax[1].set_ylabel('Energy Period [s]')
    plt.savefig('fig/wave_stats.imu.png')

    ax = plt.figure(figsize=(10,7)).add_axes([.14, .14, .8, .74])
    ax.scatter(t, direction, label='Wave direction (towards)')
    ax.scatter(t, spread, label='Wave spread')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax.set_ylabel('deg')
    plt.legend()
    plt.savefig('fig/wave_direction.imu.png')

    ds_avg = xr.Dataset()
    ds_avg['Szz'] = Szz
    ds['rpa'] = rpa
    ds_avg['Srr'] = Sxx
    ds_avg['Spp'] = Syy
    ds_avg['Saa'] = Saa
    ds_avg['Hs'] = Hs.to_xarray()['Hm0']
    ds_avg['Tm'] = Tm.to_xarray()['Tm']
    ds_avg['Te'] = Te.to_xarray()['Te']
    ds_avg['Tp'] = Tp.to_xarray()['Tp']
    ds_avg['Cra'] = Cxz
    ds_avg['Cpa'] = Cyz
    ds_avg['a1'] = a
    ds_avg['b1'] = b
    ds_avg['direction'] = xr.DataArray(direction, dims=['time'])
    ds_avg['spread'] = xr.DataArray(spread, dims=['time'])

    plt.figure(figsize=(10,7))
    plt.plot(ds.time, ds['roll'], label='roll')
    plt.plot(ds.time, ds['pitch'], label='pitch')
    plt.ylim((-35, 35))
    plt.legend()
    plt.savefig('fig/pitch_roll.imu.png')

    tilt = calc_tilt(ds['roll']-ds['roll'].mean(),
                     ds['pitch']-ds['pitch'].mean(),)
    tilt_med = tilt.rolling(time=30, center=True).median()

    plt.figure(figsize=(10,7))
    plt.plot(ds.time, tilt, label='tilt')
    plt.plot(ds.time, tilt_med, label='median filter')
    plt.ylim((0, 35))
    plt.legend()
    plt.savefig('fig/tilt.imu.png')

    return ds_avg


if __name__=='__main__':
    # Buoy deployment config
    fs = 4 # Hz
    nbin = int(fs*600) # 10 minute FFTs
    filt_freq = 0.0455  # max 22 second waves

    files = glob(os.path.join('rPi','*_IMU*.dat'))
    ds = read_yost_imu(files[0])
    for i in range(len(files)-1): # don't read last file if not complete
        ds = xr.merge((ds, read_yost_imu(files[i])))

    ds.attrs['fs'] = fs
    ds = transform_data(ds)
    ds = filter_accel(ds, filt_freq)

    # time_slc = slice(np.datetime64('2023-07-29 00:09:30'),
    #                  np.datetime64('2023-07-29 01:21:00'))
    # ds = ds.sel(time=time_slc)
    ds_avg = process_data(ds, nbin, fs)

    #ds.to_netcdf('rPi_imu.nc')
    #ds_avg.to_netcdf('rPi_imu_spec.nc')
