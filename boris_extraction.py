
import numpy as np
import os
import pandas as pd

def threshold_bouts(start_stop_array, min_iti, min_bout):
    """
    thresholds behavior bouts
    by combining behavior bouts with interbout intervals of < min_iti
    and then removing remaining bouts of < min_bout
    
    Args (3):
        start_stop_array: numpy array of dim (# of bouts, 2)
        min_iti: float, min interbout interval in seconds
        min_bout: float, min bout length in seconds
    
    Returns (1): 
        start_stop_array: numpy array (ndim=(n bouts, 2)) of start&stop times (ms)
    """
    
    start_stop_array = np.sort(start_stop_array.flatten())
    times_to_delete = []
    if min_iti > 0:
        for i in range(1, len(start_stop_array)-1, 2):
            if (start_stop_array[i+1] - start_stop_array[i]) < min_iti:
                times_to_delete.extend([i, i+1])
    start_stop_array = np.delete(start_stop_array, times_to_delete)
    bouts_to_delete = []
    if min_bout > 0:
        for i in range(0, len(start_stop_array)-1, 2):
            if start_stop_array[i+1] - start_stop_array[i] < min_bout:
                bouts_to_delete.extend([i, i+1])
    start_stop_array = np.delete(start_stop_array, bouts_to_delete)
    no_bouts = len(start_stop_array)/2
    start_stop_array = np.reshape(start_stop_array, (int(no_bouts),2))
    
    return start_stop_array
                                        
def get_behavior_bouts(boris_df, subject, behavior, min_iti = 0, min_bout = 0):
    #to do: make sure these if loops work correctly, also make subject a list always
    """
    extracts behavior bout start and stop times from a boris df
    thresholds individually by subject and behavior
    returns start_stop_array ordered by start values
    
    Args (5 total, 3 required):
        boris_df: pandas dataframe of a boris file (aggregated event table)
        subject: string or list of strings, desired subject(s) (as written in boris_df)
        behavior: string or list of strings, desired behavior(s) (as written in boris_df)
        min_iti: float, default=0, bouts with itis (s) < min_iti will be combined
        min_bout: float, default=0, bouts of length < threshold (s) will be deleted
        

    Returns (1):
        numpy array (ndim=(n bouts, 2)) of start&stop times (ms)
    """
    start_stop_arrays = []
    if type(subject) != list:
        subject = [subject]
    if type(behavior) != list:
        behavior = [behavior]
    for mouse in subject:
        subject_df = boris_df[boris_df['Subject'] == mouse]
        for act in behavior:
            behavior_df = subject_df[subject_df['Behavior'] == act]
            start_stop_array = behavior_df[['Start (s)', 'Stop (s)']].to_numpy()
            start_stop_arrays.append(threshold_bouts(start_stop_array, min_bout, min_iti))          
    start_stop_array = np.concatenate(start_stop_arrays)
    organizer = np.argsort(start_stop_array[:,0])
    start_stop_array = start_stop_array[organizer]
    
    return start_stop_array * 1000

def save_behavior_bouts(directory, boris_df, subject, behavior, min_bout = 0, min_iti = 0, filename = None):
    """
    saves a numpy array of start&stop times (ms)
    as filename: subject_behavior_bouts.npy
    
    
    Args (7 total, 4 required):
        directory: path to folder where filename.npy will be saved
            path format: './folder/folder/'
        boris_df: pandas dataframe of a boris file (aggregated event table)
        subject: string or list of strings, desired subject(s) (as written in boris_df)
        behavior: string or list of strings, desired behavior(s) (as written in boris_df)
        min_iti: float, default=0, bouts with itis (s) < min_iti will be combined
        min_bout: float, default=0, bouts of length < threshold (s) will be deleted
        filename: string, default=None, must end in .npy
    
    """
    bouts_array = get_behavior_bouts(boris_df, threshold, subject, behavior, sampling_rate)
    if filename == None:
        if type(subject) == list:
            subject = '_'.join(subject)
        if type(behavior) == list:
            behavior = '_'.join(behavior)
        subject = subject.replace(" ", "")
        behavior = behavior.replace(" ","")
        filename = f"{subject}_{behavior}_bouts.npy"
        
    np.save(path+filename, bouts_array)
    
