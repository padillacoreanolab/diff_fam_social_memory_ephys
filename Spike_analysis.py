import numpy as np
import os
import csv
import math


def get_spike_specs(path):
    
    """
    imports a spike_time and spike_unit from phy folder
    deletes spikes from units labeled noise in unit and timestamp array
    
    Args (1):
        path: path to phy folder
        path format: './folder/folder/phy'
    
    Returns (2):
        timestamps: numpy array of spike timestamps relative to sampling rate
        unit_array: numpy array of unit ids relative to timestamp array 
    """
    
    timestamps = 'spike_times.npy'
    unit = 'spike_clusters.npy'
    labels_dict = get_unit_labels(path)
    
    timestamp_array = np.load(os.path.join(path, timestamps))
    unit_array = np.load(os.path.join(path, unit))
   
    spikes_to_delete = []
   
    for spike in range(len(timestamp_array)): 
        if labels_dict[unit_array[spike].astype(str)] == 'noise':
                spikes_to_delete.append(spike)
        if unit_array[spike] in timestamp_dict.keys():
            timestamp_list = timestamp_dict[unit_array[spike]] 
            timestamp_list = np.append(timestamp_list, spiketrain_array[spike])
            timestamp_dict[unit_array[spike]] = spikestamp_list
        else:
            timestamp_dict[unit_array[spike]] = spikestamp_array[spike]
        if spike != 0:
            if timestamps_array[spike - 1] == spikestamps_array[spike]:
                print('2 spikes in a sampling')
                
    timestamps_array = np.delete(timestamps_array, spikes_to_delete)
    unit_array = np.delete(unit_array, spikes_to_delete)
    
    return timestamps, unit_array

def get_unit_labels(path):
    """
    creates a dictionary with unit id as key and label as value
    labels: good, mua, noise 
    
    Arg (1):
        path: path to phy folder
        
    Return (1):
        labels_dict: dict, unit ids, int(?): label, string
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
    
    Arg(2):
        timestamp_array: numpy array, spike timestamps
        unit_array: numpy array, unit ids associated with spikes
    
    Return (1):
        timestamp_dict: dict, unit id as keys, spike timestamps (numpy array) as values 
    """
    #thoughts: does this make more sense to have a single argument and it just be the path to the phy folder? 
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
    
    Args (3, 2 required):
        timestamp_array: numpy array, spike timestamp array
        recording_length: int, length of recording (min)
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Returns:
        spiketrain_ms_timebins: a numpy array 
            array elements are number of spikes per ms 
    """
    
    #recording_length = recording_length * 60 * sampling_rate
    hz_to_ms = int(sampling_rate*.001)
    spiketrain = np.zeros(recording_length + 1)
    for i in range(len(timestamp_array)):
        spiketrain[timestamp_array[i]] = 1
        
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
        
def get_firing_rate(timestamp_array, recording_length, smoothing_window = 250, timebin = 1, sampling_rate = 20000):
    """
    calculates firing rate (spikes/second)
    
    Args (5, 2 required):
        timestamp_array: numpy array, spike timestamp array
        recording_length: int, length of recording (min)
        smoothing_window: int, default = 250, smoothing average window (ms)
            min smoothing_window = 1
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Return (1):
        firing_rate: numpy array of firing rates in 1 ms time bins
        
    """ 
    spiketrain = get_spiketrain(timestamp_array, recording_length, sampling_rate)
    
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
    #to do: figure out how unit ids are saved into dictionaries (i.e. labels_dict and spike_dict) are they ints or strings
    """
    Creates a dictionary of spiketrains per good unit 
    where unit id is key and unit spike train is value
    does not create spiketrains for mua's
    
    Args (4, 3 required):
        labels_dict: dict, unit id for keys and labels for values
        spike_dict: dict, unit id for keys and spike timestamps for values
        recording_length: int, recording length in min
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Reutrn (1):
        unit_spiketrains: dict, unit ids as keys and unit spiketrains (in 1 ms timebins)
        
    """
    unit_spiketrains = {}
    
    for unit in spike_dict.keys():
        if labels_dict[unit] == 'good':
            unit_spiketrains[unit] = get_spiketrain(spike_dict[unit], recording_length, sampling_rate)
            
    return unit_spiketrains

def get_unit_firing_rates(labels_dict, spike_dict, recording_length, smoothing_window = 250, timebin = 1, sampling_rate = 20000):  
    """
    Calculates firing rates per unit 
    
    Args (6, 3 required):
        labels_dict:dict, unit id for keys and labels for values
        spike_dict: dict, unit id for keys and spike timestamps for values
        recording_length: int, recording length in min
        smoothing_window: int, default = 250, smoothing average window (ms)
            min smoothing_window = 1 
        timebin: int, default 1, timebin in ms for firing rate array
        sampling_rate: int, default=20000, sampling rate of ephys recording
        
    Return (1):
        unit_firing_rates: dict, unit ids as keys and unit firing rates as values
    """
    unit_firing_rates = {}
    for unit in spike_dict.keys():
        if labels_dict[unit] == 'good':
            unit_firing_rates[unit] = get_firing_rate(spike_dict[unit], smoothing_window, timebin, sampling_rate)
        
    return unit_firing_rates