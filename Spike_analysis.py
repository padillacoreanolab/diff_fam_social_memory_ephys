import numpy as np
import os
import csv
import math


def get_spike_specs(path):
    
    """
    imports a spike_time and spike_unit from phy folder
    deletes spikes from units labeled noise in unit and timestamp array
    
    Args (1 total):
        path: path to phy folder
        path format: './folder/folder/phy'
    
    Returns (2):
        timestamps: numpy array of spike timestamps relative to sampling rate
        unit_array: numpy array of unit ids relative to timestamp array 
    """
    
    timestamps = 'spike_times.npy'
    unit = 'spike_clusters.npy'
    labels_dict = get_unit_labels(path)
    
    timestamps = np.load(os.path.join(path, timestamps))
    unit_array = np.load(os.path.join(path, unit))
   
    spikes_to_delete = []
   
    for spike in range(len(timestamps)): 
        if labels_dict[unit_array[spike].astype(str)] == 'noise':
                spikes_to_delete.append(spike)
        if spike != 0:
            if timestamps[spike - 1] == timestamps[spike]:
                print('2 spikes in a sampling')
                
    timestamps = np.delete(timestamps, spikes_to_delete)
    unit_array = np.delete(unit_array, spikes_to_delete)
    
    return timestamps, unit_array

def get_unit_labels(path):
    """
    creates a dictionary with unit id as key and label as value
    labels: good, mua, noise 
    
    Arg (1 total):
        path: path to phy folder
        
    Return (1):
        labels_dict: dict, unit ids (str): label (str)
    """
    
    labels = 'cluster_group.tsv'
    with open(os.path.join(path, labels), 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        labels_dict = {row['cluster_id']: row['group'] for row in reader}
        
    return labels_dict

def get_unit_timestamps(timestamp_array, unit_array):
    """
    creates a dictionary of units to spikes
    values are spike timestamp arrays associated with the unit as key
    no noise units are included
    
    Arg(2 total):
        timestamp_array: numpy array, spike timestamps
        unit_array: numpy array, unit ids associated with spikes
    
    Return (1):
        timestamp_dict: dict, unit id (int) as keys, spike timestamps (numpy array) as values 
    """
    
    timestamp_dict = {}
    for spike in range(len(timestamp_array)): 
        if unit_array[spike] in timestamp_dict.keys():
            timestamp_list = timestamp_dict[unit_array[spike]] 
            timestamp_list = np.append(timestamp_list, timestamp_array[spike])
            timestamp_dict[unit_array[spike]] = timestamp_list
        else:
            timestamp_dict[unit_array[spike]] = timestamp_array[spike]
    
    return timestamp_dict

    
def get_spiketrain(timestamp_array, recording_length, sampling_rate = 20000):
    """
    creates a spiketrain of ms time bins 
    each array element is the number of spikes recorded per ms
    if recording (timestamp_array) is longer than the recording_length given,
    the amount of dropped data will be printed in a message
    
    
    Args (3 total, 2 required):
        timestamp_array: numpy array, spike timestamp array
        recording_length: int, length of recording (min)
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Returns:
        spiketrain_ms_timebins: a numpy array 
            array elements are number of spikes per ms 
    """
    
    recording_length = recording_length * 60 * sampling_rate
    hz_to_ms = int(sampling_rate*.001)
    spiketrain = np.zeros(recording_length + 1)
    for i in range(len(timestamp_array)):
        try:
            spiketrain[timestamp_array[i]] = 1
        except IndexError:
            dropped_recording = (timestamp_array[-1] - recording_length)/20000
            if dropped_recording > 0: 
                print(f"you are dropping {dropped_recording} seconds of recorded data")
            else:
                print("recording length is shorter than specified")
            break
        
    spiketrain_ms_timebins = np.zeros(math.floor(len(spiketrain)/hz_to_ms)) 
    ms_bin = 0
    
    for i in range(0, len(spiketrain), hz_to_ms):
        try:
            spiketrain_ms_timebins[ms_bin] = sum(spiketrain[i:i+hz_to_ms])
            ms_bin += 1
        except IndexError:
            #should i pad?
            pass

    return spiketrain_ms_timebins
        
def get_firing_rate(spiketrain, smoothing_window = 250, timebin = 1):
    """
    calculates firing rate (spikes/second)
    
    Args (3 total, 1 required):
        spiketrain: numpy array, in 1 ms time bin
        recording_length: int, length of recording (min)
        smoothing_window: int, default = 250, smoothing average window (ms)
            min smoothing_window = 1
        timebin: int, default = 1, window (ms) of firing rates returned

    Return (1):
        firing_rate: numpy array of firing rates in timebin sized windows
        
    """ 
    
    if timebin != 1:
        current_timebin = 0
        temp_spiketrain = np.zeros(math.ceil(len(spiketrain)/timebin))
        for i in range(0, len(spiketrain), timebin):
            try:
                temp_spiketrain[current_timebin] = sum(spiketrain[i:i+timebin])
                current_timebin += 1
            except IndexError:
                #should i pad here or just drop it? 
                temp_spiketrain[current_timebin] = sum(spiketrain[i:])
        spiketrain = temp_spiketrain
    firing_rate = np.empty(len(spiketrain) - smoothing_window)
    rolling_sum = sum(spiketrain[0:smoothing_window])
    for i in range(len(firing_rate)):
        firing_rate[i] = rolling_sum / (smoothing_window * .001 * timebin)
        rolling_sum = rolling_sum - spiketrain[i] + spiketrain[i+smoothing_window]
        
    return firing_rate

def get_unit_spiketrains(labels_dict, spike_dict, recording_length, sampling_rate = 20000):  
    """
    Creates a dictionary of spiketrains per good unit 
    where unit id is key and unit spike train is value
    does not create spiketrains for mua's
    
    Args (4 total, 3 required):
        labels_dict: dict, unit id for keys and labels for values
        spike_dict: dict, unit id for keys and spike timestamps for values
        recording_length: int, recording length in min
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Reutrn (1):
        unit_spiketrains: dict, unit ids as keys and unit spiketrains (in 1 ms timebins)
        
    """
    unit_spiketrains = {}
    
    for unit in spike_dict.keys():
        if labels_dict[str(unit)] == 'good':
            unit_spiketrains[unit] = get_spiketrain(spike_dict[unit], recording_length, sampling_rate)
            
    return unit_spiketrains

def get_unit_firing_rates(labels_dict, spike_dict, recording_length, smoothing_window = 250, timebin = 1, sampling_rate = 20000):  
    """
    Calculates firing rates per unit 
    
    Args (6 total, 3 required):
        labels_dict:dict, unit id for keys and labels for values
        spike_dict: dict, unit id for keys and spike timestamps for values
        recording_length: int, recording length in min
        smoothing_window: int, default = 250, smoothing average window (ms)
            min smoothing_window = 1
        timebin: int, default = 1, window (ms) of firing rates returned
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Return (1):
        unit_firing_rates: dict, unit ids as keys and unit firing rates as values
    """
    unit_firing_rates = {}
    for unit in spike_dict.keys():
        if labels_dict[str(unit)] == 'good':
            spiketrain = get_spiketrain(spike_dict[unit], recording_length, sampling_rate)
            unit_firing_rates[unit] = get_firing_rate(spiketrain, smoothing_window, timebin)
        
    return unit_firing_rates

def get_event_spiketrains(events, timestamp_array, pre_window, post_window, recording_length, sampling_rate):
    event_spiketrains = []
    spiketrain = get_spiketrain(timestamp_array, recording_length, sampling_rate)
    pre_window_ms = pre_window*1000
    post_window_ms = post_window*1000
    longest_event = 0
    event_lengths = []
    for i in range(events.shape[0]):
        event_length = int(events[i][1] - events[i][0])
        event_lengths.append(event_length)
        if event_length > longest_event:
            longest_event = event_length
    for i in range(events.shape[0]):
        event_diff = int(longest_event - event_lengths[i])
        pre_event = int(events[i][0] - pre_window_ms)
        post_event = int(events[i][1] + post_window_ms + event_diff)
        event_spiketrain = spiketrain[pre_event:post_event]
        event_spiketrains.append(event_spiketrain)
    return event_spiketrains

def get_event_firing_rates(events, timestamp_array, recording_length, smoothing_window = 250, timebin = 1,
                           sampling_rate = 20000, pre_window = 0, post_window = 0, equalize = False):
    """
    calculates firing rates for events
    
    Args (9 total, 3 required):
        events:numpy array of [[start (ms), stop (ms)] x n events]
        timestamp_array: numpy array of spike timestamps in 1 ms windows 
        recording_length: int, length of recording (min)
        smoothing_window: int, default = 250, smoothing average window (ms)
            min smoothing_window = 1 
        timebin: int, default 1, timebin in ms for firing rate array
        sampling_rate: int, default=20000, sampling rate of ephys recording
        pre_window: int, default = 0, seconds prior to start of event returned
        post_window: int, default = 0, seconds after end of event returned
        equalize: Boolean, default False, if True, equalizes lengths of each event to longest event
        
    Return (1):
        event_firing_rates: lst of numpy arrays of firing rates 
    """
    event_firing_rates = []
    spiketrain = get_spiketrain(timestamp_array, recording_length, sampling_rate)
    pre_window_ms = pre_window*1000
    post_window_ms = post_window*1000
    if equalize:
        longest_event = 0
        event_lengths = []
        for i in range(events.shape[0]):
            event_length = int(events[i][1] - events[i][0])
            event_lengths.append(event_length)
            if event_length > longest_event:
                longest_event = event_length
    for i in range(events.shape[0]):
        if equalize:
            event_diff = int(longest_event - event_lengths[i])
        else:
            event_diff = 0
        pre_event = int(events[i][0] - pre_window_ms)
        post_event = int(events[i][1] + post_window_ms + event_diff)
        event_spiketrain = spiketrain[pre_event:post_event]
        event_firing_rate = get_firing_rate(event_spiketrain)
        event_firing_rates.append(event_firing_rate)
    return event_firing_rates

def wilcoxon_average_firingrates(spike_dict, events, baseline_window, recording_length, max_event=0, 
                                 smoothing_window = 250, timebin = 1, sampling_rate = 20000):
    """
    calculates the wilcoxon signed-rank test for average firing rates of two windows: event vs baseline
    baseline used is an amount of time immediately prior to the event
    wilcoxon signed-rank test is applied to two sets of measurements:
    average firing rate per event, average firing rate per baseline
    
    Args (8 total, 3 required):
        spike_dict: dict, keys are unit ids, and values is a numpy array of timestamps
        events:numpy array of [[start (ms), stop (ms)] x n events]
        baseline_window: int, length of baseline firing rate (s)
        recording_length: int, length of recording (min)
        max_event: int, default=0, max length of an event (s)
        smoothing_window: int, default = 250, smoothing average window (ms)
            min smoothing_window = 1 
        timebin: int, default 1, timebin in ms for firing rate array
        sampling_rate: int, default=20000, sampling rate of ephys recording

    Return (1):
        wilcoxon_df: pandas dataframe, columns are unit ids, row[0] is wilcoxon 
    
    """
    trimmed_events = np.zeros(events.shape)
    preevent_baseline = np.zeros(events.shape)
    for bout in range(len(events)):
        if max_event != 0:
            if events[bout][1] - events[bout][0] > (max_event*1000):
                trimmed_events[bout][1] = events[bout][0]+(max_event*1000)
                trimmed_events[bout][0] = events[bout][0]
            else:
                trimmed_events[bout] = events[bout]
        preevent_baseline[bout] = [(events[bout][0] - (baseline_window*1000)+1), (events[bout][0]-1)]
    unit_averages = {}
    for unit in unit_dict:
        event_firing_rates = get_event_firing_rates(trimmed_events, spike_dict[unit], 
                                                    recording_length, smoothing_window, timebin, sampling_rate)
        preevent_firing_rates = get_event_firing_rates(preevent_baseline, spike_dict[unit], 
                                                        recording_length, smoothing_window, timebin, sampling_rate)
        event_averages = []
        preevent_averages = []
        for event in range(len(event_firing_rates)):
            event_averages.append(mean(event_firing_rates[event]))
            preevent_averages.append(mean(preevent_firing_rates[event]))
        unit_averages[unit] = [event_averages, preevent_averages]
    wilcoxon_stats = {}
    for unit in unit_averages.keys(): 
        wilcoxon_stats[unit] = wilcoxon(unit_averages[unit][0], unit_averages[unit][1], method = 'approx')
    
    wilcoxon_df = pd.DataFrame.from_dict(wilcoxon_stats)
    return wilcoxon_df