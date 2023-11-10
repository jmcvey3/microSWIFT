import os
import numpy as np
import pandas as pd
import xarray as xr
from glob import glob
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mpldt
plt.rcParams.update({'font.size': 14})

from mhkit import wave
import dolfyn


slc_freq = slice(0.0455, 1)

def read_gps(file):
    # Read GPS files
    df = pd.read_csv(file, header=None, parse_dates=True, infer_datetime_format=True)

    # Create time array - issue if the day rolls over
    day = np.datetime64(datetime.strptime(file[-23:-14], '%d%b%Y'), 'D')
    time = np.empty(np.shape(df[0]), dtype='datetime64[ms]')
    for i in range(len(df[0])):
        time[i] = day.astype(str) + ' ' + df[0][i]

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
                  'time': ('time', time)})
    return ds


def process_data(ds, nbin, fs):
    ## Using dolfyn to create spectra
    fft_tool = dolfyn.adv.api.ADVBinner(nbin, fs, n_fft=nbin, n_fft_coh=nbin)
    Sxx = fft_tool.calc_psd(ds['vel'][0], freq_units='Hz').sel(freq=slc_freq)
    Syy = fft_tool.calc_psd(ds['vel'][1], freq_units='Hz').sel(freq=slc_freq)
    Szz = (Sxx + Syy) / (2*np.pi*Sxx['freq'])**2 # deep water approx
    pd_Szz = Szz.T.to_pandas()

    uvz = xr.DataArray(np.array((ds['vel'][0].data, ds['vel'][1].data, ds['z'].data)),
                    dims=['dirUVZ','time'],
                    coords={'dirUVZ': ['u','v','z'],
                            'time': ds.time})
    csd = fft_tool.calc_csd(uvz, freq_units='Hz').sel(coh_freq=slc_freq)
    ## Wave direction and spread
    Cxz = csd.sel(C='Cxz').real
    Cyz = csd.sel(C='Cyz').real

    ## Wave analysis using MHKiT
    Hs = wave.resource.significant_wave_height(pd_Szz)
    Tm = wave.resource.average_wave_period(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)
    Tp = wave.resource.peak_period(pd_Szz)

    a = Cxz.values/np.sqrt((Sxx+Syy)*Szz)
    b = Cyz.values/np.sqrt((Sxx+Syy)*Szz)
    theta = np.arctan(b/a) # degrees CCW from East, "to" convention
    phi = np.sqrt(2*(1 - np.sqrt(a**2 + b**2)))
    phi = np.nan_to_num(phi) # fill missing data

    direction = np.arange(len(Szz.time))
    spread = np.arange(len(Szz.time))
    for i in range(len(Szz.time)):
        # degrees CW from North, "to" convention (90 - X)
        # degrees CW from North, "from" convention (-90 - X)
        direction[i] = -90 - np.rad2deg(np.trapz(theta[i], Szz.freq))
        spread[i] = np.rad2deg(np.trapz(phi[i], Szz.freq))


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
    plt.savefig('fig/wave_spectrum.gps.png')

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
    fig.savefig('fig/wave_stats.gps.png')

    ax = plt.figure(figsize=(10,7)).add_axes([.14, .14, .8, .74])
    ax.scatter(t, direction, label='Wave direction (towards)')
    ax.scatter(t, spread, label='Wave spread')
    ax.set_xlabel('Time')
    ax.xaxis.set_major_formatter(mpldt.DateFormatter('%D %H:%M:%S'))
    ax.set_ylabel('deg')
    plt.legend()
    plt.savefig('fig/wave_direction.gps.png')

    ds_avg = xr.Dataset()
    ds_avg['Suu'] = Sxx
    ds_avg['Svv'] = Syy
    ds_avg['Szz'] = Szz
    ds['uvz'] = uvz
    ds_avg['Hs'] = Hs.to_xarray()['Hm0']
    ds_avg['Tm'] = Tm.to_xarray()['Tm']
    ds_avg['Te'] = Te.to_xarray()['Te']
    ds_avg['Tp'] = Tp.to_xarray()['Tp']
    ds_avg['Cuz'] = Cxz
    ds_avg['Cvz'] = Cyz
    ds_avg['a1'] = a
    ds_avg['b1'] = b
    ds_avg['direction'] = xr.DataArray(direction, dims=['time'])
    ds_avg['spread'] = xr.DataArray(spread, dims=['time'])

    fig, ax = plt.subplots(figsize=(10,7))
    ax.scatter(ds["lon"], ds["lat"])
    ax.set(ylabel="Latitude [deg N]", xlabel="Longitude [deg E]")
    ax.ticklabel_format(axis='both', style='plain', useOffset=False)
    fig.savefig('fig/location.gps.png')

    return ds_avg


if __name__=='__main__':
    # Buoy deployment config
    fs = 4 # Hz
    nbin = int(fs*600) # 10 minute FFTs

    files = glob(os.path.join('NPG_4697186','*_GPS*.dat'))
    ds = read_gps(files[0])
    for i in range(len(files)):
        ds = xr.merge((ds, read_gps(files[i])))
    ds.attrs['fs'] = fs

    # time_slc = slice(np.datetime64('2023-07-29 00:09:30'),
    #                  np.datetime64('2023-07-29 01:21:00'))
    # ds = ds.sel(time=time_slc)
    ds_avg = process_data(ds, nbin, fs)

    #ds.to_netcdf('rPi_gps.nc')
    #ds_avg.to_netcdf('rPi_gps_spec.nc')
