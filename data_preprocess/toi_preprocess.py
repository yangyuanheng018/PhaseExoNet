import os
import pandas as pd
import numpy as np
import matplotlib.pylab as plt

from astropy.io import fits
from astropy.io import fits
from wotan import flatten

from process_lightcurve_with_two_cadence import process_lightcurves_into_input_representation
from preprocess import lightcurve_detrending

def process_toi_lightcurve_into_numpy_array(hdulist, info, replace_outliers=True):
    
    """The function is to get the time series and the flux series from the toi light curve file."""

    ## get planetary and stellar parameters
    quality = hdulist[1].data['QUALITY']

    ## extract data from tces.csv and light curve
    tess_mag = hdulist[0].header['TESSMAG']
    star_teff = hdulist[0].header['TEFF']
    star_logg = hdulist[0].header['LOGG']
    star_rad = info['Star Radius Value']
    star_mass = None  # 0.
    epoch = float(info['Epoch Value'])
    period = float(info['Orbital Period Value'])
    duration = float(info['Transit Duration Value']) / 24.  # unit days
    depth = float(info['Transit Depth Value'])

    if star_rad == '':
        star_rad = None
    '''if star_mass == '':
        star_mass = None'''

    ## get time and flux
    time = hdulist[1].data['time']
    flux = hdulist[1].data['PDCSAP_FLUX']

    ## adjust epoch to first transit
    epoch = np.mod(epoch - time[0], period) + time[0]
    
    ## process outliers
    if replace_outliers:
        quality_flag = np.where(np.array(quality) != 0)
        valid_ind = np.where(np.array(quality) == 0)
        median_flux = np.median(flux[valid_ind])
        flux[quality_flag] = median_flux
        
    ## detrend and remove outliers
    time, flux = lightcurve_detrending(time, flux, period, epoch, duration)
   
    ## adjust the time series to a fixed length   
    if len(time) < 20076 or len(flux) < 20076:
        flux = np.pad(flux, (0, 20076-len(flux)), 'constant', constant_values=(0, np.nan))
        time_pad = [time[-1]+i*2./(60.*24) for i in range(1, 20076-len(time)+1)]
        time = np.append(time, time_pad)
    else:
        flux = flux[:20076]
        time = time[:20076]
    # print(len(time), len(flux))
    return time, flux, [star_rad, star_mass, tess_mag, star_teff, star_logg, epoch, period, duration, depth]


if __name__ == '__main__':
    csv_filename = '../target_info/tois.csv'
    lightcuvre_dir = '../data/lc_data/tois_lc_data_test/'
    Output_Path = '../model_input/'

    ## read tois.csv
    data = pd.read_csv(csv_filename, header=0, skiprows=4)

    ## Start reading and processing light curve
    toi_num = 0

    ## folded lightcurves and other parameters
    tic = []
    lcs = []
    transits = []
    pds = []
    stellar_params = []
    dispositions = []
    _tmag, _period, _duration, _depth, _snr = [], [], [], [], []
    
    for index, info in data.iterrows():
        sector, ticid, disposition = int(info['Sectors'][0:2]), int(info['TIC']), info['TOI Disposition']
        if pd.isna(info['Orbital Period Value']) or pd.isna(info['Epoch Value']) or pd.isna(info['Transit Duration Value']):
            continue
        try:
            hdulist = fits.open(lightcuvre_dir + '{:016d}.fits'.format(ticid))
            
            ## extract data from original lightcurve file
            time, flux, params = process_toi_lightcurve_into_numpy_array(hdulist, info)
            
            padded_flux, obs_transits, period = process_lightcurves_into_input_representation(time, flux, params[6], params[5])
            
            ## save data to npy
            tic.append(ticid)
            lcs.append(padded_flux)
            transits.append(obs_transits)
            pds.append(period)
            stellar_params.append([params[0], params[1], params[2], params[3], params[4]])
            dispositions.append(disposition)
            
            _tmag.append(info['TMag Value'])
            _period.append(info['Orbital Period Value'])
            _duration.append(info['Transit Duration Value'])
            _depth.append(info['Transit Depth Value'])
            _snr.append(info['Signal-to-noise'])
            
            toi_num += 1 
        except:
            continue

    tic = np.array(tic)
    lcs = np.array(lcs)
    transits = np.array(transits)
    pds = np.array(pds)
    stellar_params = np.array(stellar_params, dtype=float)
    dispositions = np.array(dispositions)
    
    _tmag = np.array(_tmag)
    _period = np.array(_period)
    _duration = np.array(_duration)
    _depth = np.array(_depth)
    _snr = np.array(_snr)  

    print(f'Total Tois: {toi_num}')

    np.savez(Output_Path+'tois.npz', tic=tic, lc=lcs, transits=transits, period=pds, dispositions=dispositions)
    np.savez(Output_Path+'valid_spoc_toi_info.npz', tic=tic, tmag=_tmag, period=_period, duration=_duration, depth=_depth, snr=_snr)

