import os
import numpy as np
import pandas as pd
import matplotlib.pylab as plt

from astropy.io import fits
#import lightkurve as lk
from wotan import flatten

from process_lightcurve_with_two_cadence import process_lightcurves_into_input_representation

#np.set_printoptions(threshold=np.inf)

def get_ticid_from_sectors(sector_dir):
    """This function is to get the ticid in all sectors."""
    global all_ticids
    
    all_ticids = dict()
    for sector in range(1, 6):
        f = open(sector_dir + 'tesscurl_sector_{}_lc.sh'.format(sector))
        data = f.readlines()
        f.close()

        del data[0]
        sector_ticids = []

        for l in data:
            ticid = l.split('-')[6]
            sector_ticids.append(int(ticid))
        all_ticids[sector] = sector_ticids

def lightcurve_detrending(time, flux, period, epoch, duration):

    transits = np.arange(epoch, time[-1], period)
    lc_detrend = flatten(time,
        		  flux,
        		  method='median',
        		  window_length=1.0,
        		  edge_cutoff=0.1,
        		  break_tolerance=0.5,
        		  cval=5)
    #print(time, lc_detrend)
    '''
    plt.subplot(211)
    plt.title('{} {} '.format(ticid, disposition))
    plt.plot(time, flux, '.')
    plt.plot(transits,np.nanmin(flux)*np.ones(len(transits))+1.0,'r^')
    plt.subplot(212)
    plt.plot(time, lc_detrend, '.')
    plt.plot(transits,np.nanmin(lc_detrend)*np.ones(len(transits)),'r^')
    plt.show()
    '''
    return time, lc_detrend

def process_lightcurve_into_numpy_array(hdulist, row, replace_outliers=True):
    """The function is to get the time series and the flux series from the original light curve file."""

    ## get planetary and stellar parameters
    quality = hdulist[1].data['QUALITY']
    tess_mag = hdulist[0].header['TESSMAG']
    star_teff = hdulist[0].header['TEFF']
    star_logg = hdulist[0].header['LOGG']
    star_rad = tces_info["star_rad"][row]
    star_mass = tces_info["star_mass"][row]
    epoch = float(tces_info["Epoc"][row])
    period = float(tces_info["Period"][row])
    duration = float(tces_info["Duration"][row]) / 24.  # unit days
    depth = float(tces_info["Transit_Depth"][row])

    if star_rad == '':
        star_rad = None
        
    if star_mass == '':
        star_mass = None

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
        flux = np.pad(flux, (0, 20076 - len(flux)), 'constant', constant_values=(0, np.nan))
        time_pad = [time[-1] + i * 2. / (60. * 24) for i in range(1, 20076 - len(time) + 1)]
        time = np.append(time, time_pad)
    else:
        flux = flux[:20076]
        time = time[:20076]
    '''
    plt.plot(time,flux,'.')
    transits = np.arange(epoch, time[-1], period)
    plt.plot(transits,np.nanmin(flux)*np.ones(len(transits)),'^')
    plt.show()
    print(len(time), len(flux))
    '''
    return time, flux, [star_rad, star_mass, tess_mag, star_teff, star_logg, epoch, period, duration, depth]

def load_lightcurve(row, _BASE_PATH):
    ticid = int(tces_info["tic_id"][row])
    sector = int(tces_info["Sectors"][row])
    if ticid in all_ticids[sector]:
        try:
            hdulist = fits.open(_BASE_PATH+'sector{}/{:016d}.fits'.format(sector,ticid))
            return hdulist
        except:
            return None
    else:
        return None

def read_tces_file_and_drop_duplicates(csvfilename):
    global tces_info
    
    tces_info = pd.read_csv(csvfilename)
    print(len(tces_info))
    tces_info.drop('row_id', axis=1, inplace=True)
    print(tces_info[:5])
    tces_info.drop_duplicates(inplace=True)
    tces_info.reset_index(drop=True, inplace=True)
    print(len(tces_info))
 
def make_dataset(csvfilename, _BASE_PATH):
    
    read_tces_file_and_drop_duplicates(csvfilename)
    
    tic_ids = [] 
    lcs, pds, transits, stellar_params = [], [], [], [] 
    dispositions = []
    for row in range(len(tces_info)):
        if pd.isna(tces_info['Period'][row]) or pd.isna(tces_info['Epoc'][row]) or pd.isna(tces_info['Duration'][row]):
            print(f'TIC: {tces_info["tic_id"][row]} lacks necessary information!')
            continue
        
        hdulist = load_lightcurve(row, _BASE_PATH)
        if hdulist is None:
            continue
        
        time, flux, params = process_lightcurve_into_numpy_array(hdulist, row, replace_outliers=True)
        padded_flux, obs_transits, period = process_lightcurves_into_input_representation(time, flux, params[6],
                                                                                          params[5])
        tic_ids.append(tces_info['tic_id'][row])
        lcs.append(padded_flux)
        pds.append(period)
        transits.append(obs_transits)
        stellar_params.append([params[0], params[1], params[2], params[3], params[4]])
        dispositions.append(tces_info['Disposition'][row])
    
    dataset = dict()
    dataset["tic"] = np.array(tic_ids)
    dataset["lc"] = np.array(lcs)
    dataset["transits"] = np.array(transits)
    dataset["period"] = np.array(pds)
    dataset["stellar_params"] = np.array(stellar_params, dtype=float)
    dataset["disposition"] = np.array(dispositions)

    print(f'Total tces: {len(dataset["lc"])}')
    print(f'PC numbers: {np.sum(dataset["disposition"] == "PC")}')
    
    return dataset

def split_train_test_and_save_npzfile(dataset, Output_Path):
    ## fill the NaNs in the light curve with value of one
    for i in range(len(dataset["lc"])):
        dataset["lc"][i] = np.nan_to_num(dataset["lc"][i], copy=True, nan=1.0)
    
    ## shuffle the whole data and split
    n_data = len(dataset["tic"])
    dice = np.arange(n_data)
    np.random.shuffle(dice)
    n_train = int(n_data*0.8)
    
    print(np.sum(dataset["disposition"][dice[:n_train]]=='PC')/(0.8*n_data))
    print(np.sum(dataset["disposition"][dice[n_train:]]=='PC')/(0.2*n_data))
    
    np.savez(Output_Path+'train_80.npz', tic=dataset["tic"][dice[:n_train]], lc=dataset["lc"][dice[:n_train]], transits=dataset["transits"][dice[:n_train]], period=dataset["period"][dice[:n_train]], dispositions=dataset["disposition"][dice[:n_train]])
    np.savez(Output_Path+'test_20.npz', tic=dataset["tic"][dice[n_train:]], lc=dataset["lc"][dice[n_train:]], transits=dataset["transits"][dice[n_train:]], period=dataset["period"][dice[n_train:]], dispositions=dataset["disposition"][dice[n_train:]])
    
    print("Finish preprocessing!")


if __name__ == '__main__':
    #parser = argparse.ArgumentParser(description="Script for generating datasets for training purposes.")
    #parser.add_argument("--input", type=str, required=True, help="Input folder where the .fits file are")
    #parser.add_argument("--output", type=str, required=True)
    #args = parser.parse_args()

    csv_filename = '../target_info/tces.csv'
    _BASE_PATH = '../data/lc_data/' ## Note: you need to get the original fits file here
    sector_path = '../tess_target_pixel/sectors_lc_sh/'
    Output_Path = '../model_input/'
    
    get_ticid_from_sectors(sector_path)
    
    dataset = make_dataset(csv_filename, _BASE_PATH)
    
    split_train_test_and_save_npzfile(dataset, Output_Path)
   
