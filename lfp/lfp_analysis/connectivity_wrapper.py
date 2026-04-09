import numpy as np
from spectral_connectivity import Multitaper, Connectivity


def connectivity_wrapper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    power = calculate_power(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    coherence = calculate_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    granger = calculate_granger(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # pdc = calculate_partial_directed_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    return connectivity, frequencies, power, coherence, granger#, pdc


def calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    multi_t = Multitaper(
        # multitaper takes in a time_series that is time by signals (regions)
        time_series=rms_traces,
        sampling_frequency=downsample_rate,
        time_halfbandwidth_product=halfbandwidth,
        time_window_duration=timewindow,
        time_window_step=timestep,
    )
    connectivity = Connectivity.from_multitaper(multi_t)
    frequencies = connectivity.frequencies
    return connectivity, frequencies


def calculate_power(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # connectivity.power.() = [timebins, frequencies, signal]
    power = connectivity.power()
    print("Power Calculated")
    return power


def calculate_phase():
    return

def calculate_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # calculates a matrix of timebins, frequencies, region, region
    # such that [x,y,a,a] = nan
    # and [x,y,a,b] = [x,y,b,a] which is the coherence between region a & b
    # for frequency y at time x
    coherence = connectivity.coherence_magnitude()
    print("Coherence calcualatd")
    return coherence

# def calculate_partial_directed_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
#     connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
#     pdc = connectivity.partial_directed_coherence()
#     print('Partial Directed Coherence calculated')
#     return pdc
    

def calculate_granger(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    # calculates a matrix of timebins, frequencies, region, region
    # [x,y,i,j] -> granger j --> i
    # https://spectral-connectivity.readthedocs.io/en/latest/examples/Tutorial_Using_Paper_Examples.html
    # https://github.com/Eden-Kramer-Lab/spectral_connectivity/issues/31

    n_regions = rms_traces.shape[1]
    nan_cols = np.where(np.all(np.isnan(rms_traces), axis=0))[0]
    valid_cols = np.where(~np.all(np.isnan(rms_traces), axis=0))[0]

    if len(nan_cols) > 0:
        rms_clean = rms_traces[:, valid_cols]
    else:
        rms_clean = rms_traces

    connectivity, frequencies = calculate_multitaper(rms_clean, downsample_rate, halfbandwidth, timewindow, timestep)
    granger = connectivity.pairwise_spectral_granger_prediction()
    # granger shape: [frames, freq, n_valid, n_valid]

    if len(nan_cols) > 0:
        n_frames, n_freq = granger.shape[0], granger.shape[1]
        full_granger = np.full((n_frames, n_freq, n_regions, n_regions), np.nan)
        full_granger[np.ix_(np.arange(n_frames), np.arange(n_freq), valid_cols, valid_cols)] = granger
        granger = full_granger

    print("Granger causality calculated")
    return granger


    
