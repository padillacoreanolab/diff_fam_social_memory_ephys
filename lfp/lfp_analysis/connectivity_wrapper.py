from spectral_connectivity import Multitaper, Connectivity


def connectivity_wrapper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    power = calculate_power(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    coherence = calculate_coherence(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # granger = calculate_grangers(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    return connectivity, frequencies, power, coherence  # , granger


def calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    multi_t = Multitaper(
        # multitaper takes in a time_series that is time by signals (regions)
        time_series=rms_traces.T,
        sampling_frequency=downsample_rate,
        time_halfbandwidth_product=halfbandwidth,
        time_window_duration=timewindow,
        time_window_step=timestep,
    )
    print("sampling freq", downsample_rate)
    print("half bandwidth", halfbandwidth)
    print("duration", timewindow)
    print("step", timestep)
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


def calculate_grangers(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep):
    connectivity, frequencies = calculate_multitaper(rms_traces, downsample_rate, halfbandwidth, timewindow, timestep)
    # calculates a matrix of timebins, frequencies, region, region
    # such that [x,y,a,a] = nan
    # and [x,y,a,b] =/= [x,y,b,a]
    # [x,y,a,b] -> a to b granger?
    # [x,y,b,a] -> b to a granger?
    granger = connectivity.pairwise_spectral_granger_prediction()
    print("Granger's causality calculated")
    return granger
