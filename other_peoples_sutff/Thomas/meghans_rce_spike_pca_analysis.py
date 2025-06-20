# %%
%cd ..

# %%
import os 
import sys
import pickle
import numpy as np
# Add the project root to Python path
# sys.path.insert(0, '/blue/npadillacoreano/mcum/SynapticSync/src')
#from spike.spike_analysis.spike_collection import SpikeCollection
#from synapticsync.spike.spike_collection import SpikeCollection

# %%
def pickle_this(thing_to_pickle, file_name):
    """
    Pickles things
    Args (2):
        thing_to_pickle: anything you want to pickle
        file_name: str, filename that ends with .pkl
    Returns:
        none
    """
    with open(file_name, "wb") as file:
        pickle.dump(thing_to_pickle, file)
def unpickle_this(pickle_file):
    """
    Unpickles things
    Args (1):
        file_name: str, pickle filename that already exists and ends with .pkl
    Returns:
        pickled item
    """
    with open(pickle_file, "rb") as file:
        return pickle.load(file)

# %% [markdown]
# # Make subject dict

# %%
behavior_dicts = {}

def make_recording_to_subj_dict(data_path):
    recording_to_subject = {}
    for root, dirs, files in os.walk(data_path):
        for file in dirs:
            if file.endswith('merged.rec'):
                if file.startswith('2023'):
                    subject = str(file.split("_")[-4].replace('-','.'))
                    recording_to_subject[file] = subject
                    behavior_dicts[file] = {}
                if file.startswith('2024'):
                    subject = str(file.split("_")[-3].replace('-','.'))
                    recording_to_subject[file] = subject
                    behavior_dicts[file] = {}   
                if (file == '20230620_114347_standard_comp_to_omission_D4_subj_1-1_t1b2L_box_2_merged.rec')| (file == '20230620_114347_standard_comp_to_omission_D4_subj_1-2_t3b3L_box_1_merged.rec'):
                    subject = str(file.split("_")[-5].replace('-','.'))
                    recording_to_subject[file] = subject
                    behavior_dicts[file] = {}
                if file =='20230618_100636_standard_comp_to_omission_D2_subj_1_1_t1b2L_box2_merged.rec':
                    recording_to_subject[file] = '1.1'
                if file == '20230618_100636_standard_comp_to_omission_D2_subj_1_4_t4b3L_box1_merged.rec':
                    recording_to_subject[file] = '1.4'
                
    return recording_to_subject

subject_dict = make_recording_to_subj_dict(r"/blue/npadillacoreano/share/reward_comp_extention/Phy_rce2_rce3/phy_curation/megadataset")

# %% [markdown]
# # Make/clean event dict

# %%
event_dict = unpickle_this(r"/blue/npadillacoreano/share/reward_comp_extention/event_dict.pkl")
ms_event_dict = {}
# convert sampoling rate to miliseconds
original_keys = 'high_comp_lose', 'high_comp_win', 'low_comp_lose', 'low_comp_win', 'alone_rewarded'
# USE THIS 
for recording, inner_dict in event_dict.items():
    ms_event_dict[recording] = {}
    for name, event in inner_dict.items():
        event = np.array(event)
        if len(event.shape) > 1:
            event = event /20
            ms_event_dict[recording][name] = event
        else:
            ms_event_dict[recording][name] = np.empty((0,2))
    for events in original_keys:
        starts = ms_event_dict[recording][events]
        try:
            event_baseline = starts - 10000
        except IndexError:
            event_baseline =  np.empty((0, 2))
        ms_event_dict[recording][f'{events}_baseline'] = event_baseline
        
        
################################################################################        
for recording in ms_event_dict.keys():
    baselines = [ms_event_dict[recording]['high_comp_lose_baseline'], 
                 ms_event_dict[recording]['high_comp_win_baseline'],
                 ms_event_dict[recording]['low_comp_lose_baseline'], 
                 ms_event_dict[recording]['low_comp_win_baseline']]
    
    # Filter out empty arrays
    non_empty_baselines = [b for b in baselines if b.size > 0]
    
    if non_empty_baselines:  # Only stack if there are non-empty arrays
        all_baselines = np.vstack(non_empty_baselines)
        ms_event_dict[recording]['overall_pretone'] = all_baselines
        
for recording in ms_event_dict.keys(): 
    hc_lose = ms_event_dict[recording]['high_comp_lose']
    hc_win = ms_event_dict[recording]['high_comp_win']
    lc_win = ms_event_dict[recording]['low_comp_win']
    lc_lose = ms_event_dict[recording]['low_comp_lose']
    # Handle win events
    if len(hc_win.shape) > 1 and len(lc_win.shape) > 1:
        win = np.vstack([hc_win, lc_win])
        ms_event_dict[recording]['win'] = win
    elif hc_win.size > 1:
        ms_event_dict[recording]['win'] = hc_win
    elif lc_win.size > 1:
        ms_event_dict[recording]['win'] = lc_win
    else:
        ms_event_dict[recording]['win'] = np.array([]).reshape(0, hc_win.shape[1]) if hc_win.ndim > 1 else np.array([])
    
    # Handle lose events
    if len(hc_lose.shape) > 1 and len(lc_lose.shape) > 1:
        ms_event_dict[recording]['lose'] = np.vstack([hc_lose, lc_lose])
    elif hc_lose.size > 1:
        ms_event_dict[recording]['lose'] = hc_lose
    elif lc_lose.size > 1:
        ms_event_dict[recording]['lose'] = lc_lose
    else:
        ms_event_dict[recording]['lose'] = np.array([]).reshape(0, hc_lose.shape[1]) if hc_lose.ndim > 1 else np.array([])
    
    # Handle high_comp events
    if len(hc_lose.shape) > 1 and len(hc_win.shape) > 1:
        ms_event_dict[recording]['high_comp'] = np.vstack([hc_lose, hc_win])
    elif hc_lose.size > 1:
        ms_event_dict[recording]['high_comp'] = hc_lose
    elif hc_win.size > 1:
        ms_event_dict[recording]['high_comp'] = hc_win
    else:
        ms_event_dict[recording]['high_comp'] = np.array([]).reshape(0, hc_lose.shape[1]) if hc_lose.ndim > 1 else np.array([])
    
    # Handle low_comp events
    if len(lc_lose.shape) > 1 and len(lc_win.shape) > 1:
        ms_event_dict[recording]['low_comp'] = np.vstack([lc_lose, lc_win])
    elif lc_lose.size > 1:
        ms_event_dict[recording]['low_comp'] = lc_lose
    elif lc_win.size > 1:
        ms_event_dict[recording]['low_comp'] = lc_win
    else:
        ms_event_dict[recording]['low_comp'] = np.array([]).reshape(0, lc_lose.shape[1]) if lc_lose.ndim > 1 else np.array([])



# %%
pickle_this(ms_event_dict, '/blue/npadillacoreano/share/reward_comp_extention/ms_event_dict_complete.pkl')

# %% [markdown]
# # Make spike object

# %%
sc_object = SpikeCollection(path=r"/blue/npadillacoreano/share/reward_comp_extention/Phy_rce2_rce3/phy_curation/megadataset",
                            event_dict=ms_event_dict, # event dict from /blue/npadillacoreano/share/reward_comp_extention/event_dict.pkl
                            subject_dict=subject_dict) # 
    

# %%
#sc_object.save_collection(r"/blue/npadillacoreano/share/reward_comp_extention/Phy_rce2_rce3/spike_collection.json")


# %%
#sc_tester_object = SpikeCollection.load_collection(r"/blue/npadillacoreano/share/reward_comp_extention/Phy_rce2_rce3/spike_collection.json")

# %%
sc_object.recording_details()

# %%
import spike.spike_analysis.pca_trajectories as pcat
# from importlib import reload
# reload(pcat)
sc_object.analyze(timebin = 50, ignore_freq = 0.5, smoothing_window = 250)

win_lose_pca = pcat.avg_trajectories_pca(sc_object, event_length = 10, pre_window = 10, events = ['win', 'lose'], min_neurons = 7, d = 3)
print(win_lose_pca)
hc_lc_pca = pcat.avg_trajectories_pca(sc_object, event_length = 10, pre_window = 10, events = ['low_comp', 'high_comp'], min_neurons = 7, d= 3)
print(hc_lc_pca)
all_pca = pcat.avg_trajectories_pca(sc_object, event_length = 10, pre_window = 10, events = ['high_comp_win', 'high_comp_lose', 'low_comp_lose','low_comp_win'], min_neurons = 7, d = 3)
print(all_pca)
alone_pca = pcat.avg_trajectories_pca(sc_object, event_length = 10, pre_window = 10, events = ['win', 'lose', 'alone_rewarded'], min_neurons = 7, d = 3)

# %%
print(alone_pca)

# %%


# %%



