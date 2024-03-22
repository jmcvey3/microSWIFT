import os
from glob import glob
import numpy as np
import pandas as pd
import xarray as xr
import scipy.signal as ss
from scipy.integrate import cumtrapz
from scipy.spatial.transform import Rotation as R
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mpldt
from mhkit import wave, dolfyn

plt.rcParams.update({"font.size": 14})
plt.close("all")


slc_freq = slice(0.0455, 1)


def read_gps(file):
    # Read GPS files
    df = pd.read_csv(file, header=None, parse_dates=True, infer_datetime_format=True)

    # Create time array - issue if the day rolls over
    day = np.datetime64(datetime.strptime(file[-23:-14], "%d%b%Y"), "D")
    time = np.empty(np.shape(df[0]), dtype="datetime64[ms]")
    for i in range(len(df[0])):
        time[i] = day.astype(str) + " " + df[0][i]

    ds = xr.Dataset(
        data_vars={
            "vel": (
                ["dir", "time"],
                np.array([df[1], df[2]]),
                {"units": "m s-1", "long_name": "Velocity"},
            ),
            "z": (
                ["time"],
                np.array(df[3]),
                {"units": "m", "long_name": "Altitude", "standard_name": "altitude"},
            ),
            "lat": (
                ["time"],
                np.array(df[4]),
                {
                    "units": "degree_N",
                    "long_name": "Latitude",
                    "standard_name": "latitude",
                },
            ),
            "lon": (
                ["time"],
                np.array(df[5]),
                {
                    "units": "degree_E",
                    "long_name": "Longitude",
                    "standard_name": "longitude",
                },
            ),
        },
        coords={"dir": ("dir", ["u", "v"]), "time": ("time", time)},
    )
    return ds


def read_yost_imu(file):
    # Yost gets the XYZ coordinate system wrong: +Y and +Z are misnamed.
    # Z is actually the Y axis and Y is actually the Z
    df = pd.read_csv(file, header=None, parse_dates=True, infer_datetime_format=True)

    ds = xr.Dataset(
        data_vars={
            # This is already in m/s2, contrary to manual
            # Also appears to be already in earth coordinates
            "accel": (
                ["dir", "time"],
                np.array([df[1], df[3], df[2]]),
                {"units": "m s-2", "long_name": "acceleration vector"},
            ),
            "angrt": (
                ["dir", "time"],
                np.array([df[4], df[6], df[5]]),
                {"units": "rad s-1", "long_name": "angular velocity vector"},
            ),
            "mag": (
                ["dir", "time"],
                np.array([df[7], df[9], df[8]]),
                {"units": "gauss", "long_name": "magnetic field vector"},
            ),
            "quaternions": (
                ["q", "time"],
                np.array([df[10], df[12], df[11], df[13]]),
                {"units": "1", "long_name": "quaternion vector"},
            ),
            "check_factor": (
                ["time"],
                np.array(df[14]),
                {"units": "1", "long_name": "check factor"},
            ),
        },
        coords={
            "dir": ("dir", ["x", "y", "z"]),
            "q": ("q", [1, 2, 3, 4]),
            "time": ("time", df[0].values.astype("datetime64[ns]")),
        },
    )
    return ds


def transform_data(ds):
    # Rotate accelerometer into Earth coordinates
    r = R.from_quat(ds["quaternions"].T)
    ds["accel"].values = r.apply(ds["accel"].T.values).T

    r = R.from_quat(ds["quaternions"].T)
    rpy = r.as_euler("XYZ", degrees=True)
    ds["roll"] = xr.DataArray(
        rpy[:, 0],
        dims=["time"],
        coords={"time": ds.time},
        attrs={
            "units": "degrees",
            "long_name": "roll",
            "standard_name": "platform_roll",
        },
    )
    ds["pitch"] = xr.DataArray(
        rpy[:, 1],
        dims=["time"],
        coords={"time": ds.time},
        attrs={
            "units": "degrees",
            "long_name": "pitch",
            "standard_name": "platform_pitch",
        },
    )
    ds["heading"] = xr.DataArray(
        rpy[:, 0],
        dims=["time"],
        coords={"time": ds.time},
        attrs={
            "units": "degrees",
            "long_name": "heading",
            "standard_name": "platform_orientation",
        },
    )

    return ds


def calc_tilt(pitch, roll):
    tilt = np.rad2deg(
        np.arctan(
            np.sqrt(np.tan(np.deg2rad(roll)) ** 2 + np.tan(np.deg2rad(pitch)) ** 2)
        )
    )
    return tilt


def calc_vel_rotation(angrt, vec=[0, 0, -0.25]):
    # Calculate the induced velocity due to rotations of a point about the IMU center.

    # Motion of the IMU about the Witt's origin should be the
    # cross-product of omega (rotation rate vector) and the vector.
    #   u=dz*omegaY-dy*omegaZ,v=dx*omegaZ-dz*omegaX,w=dy*omegaX-dx*omegaY
    # where vec=[dx,dy,dz], and angrt=[omegaX,omegaY,omegaZ]
    velrot = np.array(
        [
            (vec[2] * angrt[1] - vec[1] * angrt[2]),
            (vec[0] * angrt[2] - vec[2] * angrt[0]),
            (vec[1] * angrt[0] - vec[0] * angrt[1]),
        ]
    )
    # Rotate induced velocity to earth coordinates
    r = R.from_quat(ds["quaternions"].T)
    velrot = r.apply(velrot.T).T

    return velrot


def butter_bandpass_filter(data, fc, fh, fs, order):
    normal_cutoff = [fc / (0.5 * fs), fh / (0.5 * fs)]
    # Get the filter coefficients
    b, a = ss.butter(order, normal_cutoff, btype="bandpass", analog=False)
    y = ss.filtfilt(b, a, data)
    return y


def filter_accel(ds):
    filt_freq_low = 0.0455  # max 22 second waves
    filt_freq_high = 0.5
    filt_factor = 5 / 3  # should be 5/3 for butterworth filter

    # Run band pass filter
    accel = ds["accel"].copy()
    accel = butter_bandpass_filter(
        accel,
        filt_factor * filt_freq_low,
        filt_factor * filt_freq_high,
        ds.fs,
        1,
    )

    # Integrate
    hp = np.concatenate(
        (
            np.zeros(list(accel.shape[:-1]) + [1]),
            cumtrapz(accel, dx=1 / ds.fs, axis=-1),
        ),
        axis=-1,
    )
    # Run bandpass filter
    veldat = butter_bandpass_filter(
        hp,
        filt_factor * filt_freq_low,
        filt_factor * filt_freq_high,
        ds.fs,
        1,
    )

    # Subtract moment arm from rotating buoy
    vel_rot = calc_vel_rotation(ds["angrt"].values)
    veldat -= vel_rot  # Subtract motion in Z from accelerometer motion

    # Integrate and filter again
    hp = np.concatenate(
        (
            np.zeros(list(veldat.shape[:-1]) + [1]),
            cumtrapz(veldat, dx=1 / ds.fs, axis=-1),
        ),
        axis=-1,
    )
    posdat = butter_bandpass_filter(
        hp,
        filt_factor * filt_freq_low,
        filt_factor * filt_freq_high,
        ds.fs,
        1,
    )

    ds["velacc"] = ds["accel"].copy() * 0
    ds["velacc"].values = veldat
    ds["velacc"].attrs["units"] = "m/s"
    ds["velacc"].attrs["long name"] = "velocity vector from accelerometer"

    ds["posacc"] = ds["accel"].copy() * 0
    ds["posacc"].values = posdat
    ds["posacc"].attrs["units"] = "m"
    ds["posacc"].attrs["long name"] = "position vector from accelerometer"

    return ds


def process_data(ds_imu, ds_gps, nbin, fs):
    ds_imu = ds_imu.interp(time=ds_gps.time, kwargs={"fill_value": "extrapolate"})

    ## Wave height and period
    fft_tool = dolfyn.adv.api.ADVBinner(nbin, fs, n_fft=nbin / 3, n_fft_coh=nbin / 3)
    # Sww_imu = fft_tool.power_spectral_density(ds_imu["velacc"][2], freq_units="Hz")
    # Sww_imu = Sww_imu.sel(freq=slc_freq)
    # Szz_imu = Sww_imu / (2 * np.pi * Sww_imu["freq"]) ** 2
    Szz_imu = fft_tool.power_spectral_density(ds_imu["posacc"][2], freq_units="Hz")
    Szz_imu = Szz_imu.sel(freq=slc_freq)

    Suu = fft_tool.power_spectral_density(ds_gps["vel"][0], freq_units="Hz")
    Svv = fft_tool.power_spectral_density(ds_gps["vel"][1], freq_units="Hz")
    Sww_gps = Suu + Svv
    Sww_gps = Sww_gps.sel(freq=slc_freq)
    Szz_gps = Sww_gps / (2 * np.pi * Sww_gps["freq"]) ** 2

    plt.figure()
    plt.loglog(Szz_imu.freq, Szz_imu.mean("time"), label="IMU")
    plt.loglog(Szz_gps.freq, Szz_gps.mean("time"), label="GPS")
    m = -4
    x = np.logspace(-1, 0.5)
    y = 10 ** (-5) * x**m
    plt.loglog(x, y, "--", c="black", label="f^-4")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Energy Density [m^2/Hz]")
    plt.ylim((0.00001, 10))
    plt.legend()
    plt.savefig("fig/wave_spectrum.png")

    # Wave stats
    pd_Szz = Szz_imu.T.to_pandas()
    Hs = wave.resource.significant_wave_height(pd_Szz)
    Tm = wave.resource.average_wave_period(pd_Szz)
    Te = wave.resource.energy_period(pd_Szz)
    Tp = wave.resource.peak_period(pd_Szz)

    ## Wave direction and spread
    uva = xr.DataArray(
        np.array(
            (ds_gps["vel"][0].data, ds_gps["vel"][1].data, ds_imu["accel"][2].data)
        ),
        dims=["dirUVA", "time"],
        coords={"dirUVA": ["u", "v", "accel"], "time": ds_imu.time},
    )
    psd = fft_tool.power_spectral_density(uva, freq_units="Hz")
    psd = psd.sel(freq=slc_freq)
    Suu = psd.sel(S="Sxx")
    Svv = psd.sel(S="Syy")
    Saa = psd.sel(S="Szz")

    csd = fft_tool.cross_spectral_density(uva, freq_units="Hz")
    csd = csd.sel(coh_freq=slc_freq)
    Cua = csd.sel(C="Cxz").real
    Cva = csd.sel(C="Cyz").real

    a = Cua.values / np.sqrt((Suu + Svv) * Saa)
    b = Cva.values / np.sqrt((Suu + Svv) * Saa)
    theta = np.rad2deg(np.arctan2(b, a))  # degrees CCW from East, "to" convention
    phi = np.rad2deg(np.sqrt(2 * (1 - np.sqrt(a**2 + b**2))))

    peak_idx = psd[2].argmax("freq")
    # degrees CW from North ("from" convention)
    direction = (270 - theta[:, peak_idx]) % 360
    # Set direction from -180 to 180
    direction[direction > 180] -= 360
    spread = phi[:, peak_idx]

    ds_avg = xr.Dataset()
    ds_avg["Szz"] = Szz_imu
    ds_avg["Suu"] = Suu
    ds_avg["Svv"] = Svv
    ds_avg["Saa"] = Saa
    ds_avg["Hs"] = Hs.to_xarray()["Hm0"]
    ds_avg["Tm"] = Tm.to_xarray()["Tm"]
    ds_avg["Te"] = Te.to_xarray()["Te"]
    ds_avg["Tp"] = Tp.to_xarray()["Tp"]
    ds_avg["Cua"] = Cua
    ds_avg["Cva"] = Cva
    ds_avg["a1"] = xr.DataArray(a, dims=["time", "freq"])
    ds_avg["b1"] = xr.DataArray(b, dims=["time", "freq"])
    ds_avg["direction"] = xr.DataArray(direction, dims=["time"])
    ds_avg["spread"] = xr.DataArray(spread, dims=["time"])

    t = dolfyn.time.dt642date(psd["time"])
    fig, ax = plt.subplots(2, figsize=(10, 7))
    ax[0].scatter(t, Hs)
    ax[0].set_xlabel("Time")
    ax[0].xaxis.set_major_formatter(mpldt.DateFormatter("%D %H:%M:%S"))
    ax[0].set_ylabel("Significant Wave Height [m]")

    ax[1].scatter(t, Te)
    ax[1].set_xlabel("Time")
    ax[1].xaxis.set_major_formatter(mpldt.DateFormatter("%D %H:%M:%S"))
    ax[1].set_ylabel("Energy Period [s]")
    plt.savefig("fig/wave_stats.png")

    ax = plt.figure(figsize=(10, 7)).add_axes([0.14, 0.14, 0.8, 0.74])
    ax.scatter(t, direction, label="Wave direction")
    ax.scatter(t, spread, label="Wave spread")
    ax.set_xlabel("Time")
    ax.xaxis.set_major_formatter(mpldt.DateFormatter("%D %H:%M:%S"))
    ax.set_ylabel("deg")
    plt.legend()
    plt.savefig("fig/wave_direction.png")

    fig, ax = plt.subplots(figsize=(10, 7))
    speed = np.sqrt(ds_gps["vel"][0].values ** 2 + ds_gps["vel"][1].values ** 2)
    h = ax.scatter(ds_gps["lon"], ds_gps["lat"], c=speed, cmap="Blues")
    fig.colorbar(h, ax=ax, label="Drift Speed [m/s]")
    ax.set(ylabel="Latitude [deg N]", xlabel="Longitude [deg E]")
    ax.ticklabel_format(axis="both", style="plain", useOffset=False)
    # ax.quiver(
    #     ds_gps["lon"],
    #     ds_gps["lat"],
    #     ds_gps["vel"][0],
    #     ds_gps["vel"][1],
    # )
    fig.savefig("fig/gps.png")

    return ds_avg


if __name__ == "__main__":
    # Buoy deployment config
    fs = 4  # Hz
    nbin = int(fs * 600)  # 10 minute FFTs

    # Fetch IMU data
    files = glob(os.path.join("rPi", "*_IMU*.dat"))
    ds = read_yost_imu(files[0])
    for i in range(len(files) - 1):  # don't read last file if not complete
        ds = xr.merge((ds, read_yost_imu(files[i])))
    ds.attrs["fs"] = fs
    ds = transform_data(ds)
    ds_imu = filter_accel(ds)
    # ds_imu.to_netcdf('data/rPi_imu.raw.nc')

    # Fetch GPS data
    files = glob(os.path.join("rPi", "*_GPS*.dat"))
    ds = read_gps(files[0])
    for i in range(len(files) - 1):  # don't read last file if not complete
        ds = xr.merge((ds, read_gps(files[i])))
    ds.attrs["fs"] = fs
    ds_gps = ds
    # ds_gps.to_netcdf('data/rPi_gps.raw.nc')

    # # Start all instruments on Spotter3 activation
    # time_slc = slice(np.datetime64('2023-07-29 00:09:30'),
    #                  np.datetime64('2023-07-29 01:21:00'))
    # ds_imu = ds_imu.sel(time=time_slc)
    # ds_gps = ds_gps.sel(time=time_slc)

    ds_waves = process_data(ds_imu, ds_gps, nbin, fs)

    # ds_waves.to_netcdf('data/rPi.10m.nc')
    # dolfyn.save_mat(ds_waves, 'rPi.10m.mat', datenum=False)
