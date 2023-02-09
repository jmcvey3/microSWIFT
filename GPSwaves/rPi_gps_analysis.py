import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr

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


if __name__=='__main__':
    # Buoy deployment config
    fs = 4 # Hz
    nbin = int(fs*600) # 10 minute FFTs

    files = glob(os.path.join('rPi','*_IMU*.dat'))
    ds = read_gps(files[0])
    for i in range(len(files)):
        ds = xr.merge((ds, read_gps(files[i])))

    ds.attrs['fs'] = fs

    ## Using dolfyn to create spectra
    fft_tool = dolfyn.adv.api.ADVBinner(nbin, fs, n_fft=nbin, n_fft_coh=nbin)
    Sxx = fft_tool.calc_psd(ds['vel'][0], freq_units='Hz')
    Syy = fft_tool.calc_psd(ds['vel'][1], freq_units='Hz')
    Szz = (Sxx + Syy) / (2*np.pi*Sxx['freq'])**2 # deep water approx
    Szz = Szz.sel(freq=slice(0.0455, None))
    pd_Szz = Szz.T.to_pandas()

    uvz = xr.DataArray(np.array((ds['vel'][0].data, ds['vel'][1].data, ds['z'].data)),
                    dims=['dirUVZ','time'],
                    coords={'dirUVZ': ['u','v','z'],
                            'time': ds.time})
    csd = fft_tool.calc_csd(uvz, freq_units='Hz')
    ## Wave direction and spread
    Cxz = csd.sel(C='Cxz').real
    Cyz = csd.sel(C='Cyz').real


    ## Wave analysis using MHKiT
    Hs = wave.resource.significant_wave_height(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)

    a = Cxz/np.sqrt((Sxx+Syy)*Szz)
    b = Cyz/np.sqrt((Sxx+Syy)*Szz)
    theta = np.arctan(b/a) * (180/np.pi) # degrees CCW from East
    #theta = dolfyn.tools.misc.convert_degrees(theta) # degrees CW from North
    phi = np.sqrt(2*(1 - np.sqrt(a**2 + b**2))) * (180/np.pi)
    phi = phi.fillna(0) # fill missing data

    direction = np.arange(len(Szz.time))
    spread = np.arange(len(Szz.time))
    for i in range(len(Szz.time)):
        direction[i] = np.trapz(theta[i], Szz.freq)
        spread[i] = np.trapz(phi[i], Szz.freq)

    ds_avg = xr.Dataset()
    ds_avg['Suu'] = Sxx
    ds_avg['Svv'] = Syy
    ds_avg['Szz'] = Szz
    ds['uvz'] = uvz
    ds_avg['Hs'] = Hs.to_xarray()['Hm0']
    ds_avg['Te'] = Te.to_xarray()['Te']
    ds_avg['Cuz'] = Cxz
    ds_avg['Cvz'] = Cyz
    ds_avg['a1'] = a
    ds_avg['b1'] = b
    ds_avg['direction'] = xr.DataArray(direction, dims=['time'])
    ds_avg['spread'] = xr.DataArray(spread, dims=['time'])

    ds.to_netcdf('rPi_gps.nc')
    ds_avg.to_netcdf('rPi_gps_spec.nc')
