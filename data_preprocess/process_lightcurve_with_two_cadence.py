import numpy as np
import matplotlib.pyplot as plt


def trim_transits(transits, flux, epsilon=1e-8):

    """trim the transits in the left-end and right-end that actually not observed"""

    left_cut, right_cut = 0, 0
    for t in transits:
        if np.std(flux[int(t)-5: int(t)+6])<epsilon:
            left_cut += 1
        else:
            break
    for t in transits[::-1]:
        if np.std(flux[int(t)-5: int(t)+6])<epsilon:
            right_cut += 1
        else:
            break
    #print(left_cut, right_cut)
    if right_cut > 0:
        return transits[left_cut: -right_cut]
    else:
        return transits[left_cut:]


def split_transits(transits, num_channels=12):
    n_split = int(np.ceil(len(transits)/num_channels))
    if n_split == 1:
        return [transits], n_split
    else:
        split_tran = []
        for i in range(n_split):
            split_tran.append(transits[i*num_channels:(i+1)*num_channels])
        return split_tran, n_split


def process_lightcurves_into_input_representation(time, flux, pd, ep):
    ## define size of the input data
    #time_length = 10039  ## 10039 is half of the sector time
    #mid_length = (time_length - 1) // 2
    #num_channels = 12

    time_diff = time[1:] - time[:-1]
    cadence = np.nanmean(time_diff)
    #pd = np.clip(pd, a_min=0.0, a_max=27.88333 / 2.0)

    ## fill the nan flux time-series with the median flux value
    med_flux = np.nanmedian(flux)
    flux_filled = np.nan_to_num(flux, nan=med_flux) #/ med_flux

    ## calculate each transit time position
    transits = np.arange((ep - time[0]) / cadence, (time[-1] - time[0]) / cadence, pd / cadence)
    round_transits = np.round(transits)

    ## transit time biases due to the rounding, which contain light curve shape information
    transit_diff = transits - round_transits

    ## pad the time and flux so that each cropped period has valid values
    left_padding = np.arange(time[0] - cadence, time[0] - pd, -cadence)[::-1]
    right_padding = np.arange(time[-1] + cadence, time[-1] + pd, cadence)

    padded_time = np.concatenate([left_padding, time, right_padding], axis=0)
    padded_flux = np.concatenate([np.ones_like(left_padding), flux_filled, np.ones_like(right_padding)], axis=0)

    transits = round_transits + left_padding.shape[0]
    obs_transits = trim_transits(transits, padded_flux)
    # s_transits, _ = split_transits(obs_transits)
    pd = int(np.round(pd/cadence))

    '''
    print(time)
    print(obs_transits)
    plt.plot(np.arange(len(padded_flux)),padded_flux,'.')
    plt.plot(obs_transits,min(padded_flux)*np.ones(len(obs_transits)),'^')
    plt.show()
    '''
    return padded_flux, obs_transits, pd
