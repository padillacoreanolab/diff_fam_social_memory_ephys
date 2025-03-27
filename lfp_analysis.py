
#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os

def change_dir_in_hipergator():
    # this env var is set on hipergator (SLURM is the job scheduler there)
    if os.getenv("SLURM_JOB_ID", None):
        target_dir = "diff_fam_social_memory_ephys"
        current = os.getcwd()

        while True:
            parent = os.path.dirname(current)
            # If we've reached the root directory without finding the target
            if parent == current:
                raise FileNotFoundError(f"Could not find parent directory '{target_dir}'")

            # Check if the target directory is the current parent's name
            if os.path.basename(parent) == target_dir:
                os.chdir(parent)
                return parent

            current = parent

# Usage
try:
    new_path = change_dir_in_hipergator()
    print(f"Successfully changed to directory: {new_path}")
except FileNotFoundError as e:
    print(e)


# In[4]:


import pandas as pd
import numpy as np
import lfp.lfp_analysis.LFP_collection as LFP_collection
import lfp.lfp_analysis.Analysis as LFP_analysis
import pickle


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


# In[5]:


from importlib import reload
reload(LFP_collection)


# In[7]:


import sys

# These are the usual ipython objects, including this one you are creating
ipython_vars = ['In', 'Out', 'exit', 'quit', 'get_ipython', 'ipython_vars']

# Get a sorted list of the objects and their sizes
sorted([(x, sys.getsizeof(globals().get(x))) for x in dir() if not x.startswith('_') and x not in sys.modules and x not in ipython_vars], key=lambda x: x[1], reverse=True)


# In[6]:


#data/novel_lfp

novel_lfp_json = "data/novel_lfp/lfp_collection.json"

novel_collection = LFP_collection.LFPCollection.load_collection(novel_lfp_json)


# In[ ]:


print("GOT THE NOVEL COLLECTION")
cagemate_lfp_json = "data/cagemate_lfp/lfp_collection.json"

cagemate_collection = LFP_collection.LFPCollection.load_collection(cagemate_lfp_json)


# In[ ]:


behavior_dicts = unpickle_this('pilot2/habit_dishabit_phase1/behavior_dicts.pkl')

for recording in cagemate_collection.lfp_recordings:
    subject = str(int(recording.recording_name.split('_')[0])/10)
    recording_pattern = (recording.recording_name.split('_')[0] + '_' +
                         recording.recording_name.split('_')[1] + '_' +
                         recording.recording_name.split('_')[2] + '_' +
                         'aggregated')
    recording.behavior_dict = behavior_dicts[recording_pattern]
    print(recording_pattern)
    recording.subject = subject

for recording in novel_collection.lfp_recordings:
    subject = str(int(recording.recording_name.split('_')[0])/10)
    recording_pattern = (recording.recording_name.split('_')[0] + '_' +
                         recording.recording_name.split('_')[1] + '_' +
                         recording.recording_name.split('_')[2] + '_' +
                         'aggregated')
    recording.behavior_dict = behavior_dicts[recording_pattern]
    print(recording_pattern)
    recording.subject = subject


# In[45]:


import importlib
importlib.reload(LFP_analysis)
cagemate_analysis = LFP_analysis.LFPAnalysis(cagemate_collection.lfp_recordings)
novel_analysis = LFP_analysis.LFPAnalysis(novel_collection.lfp_recordings)


# In[46]:


events = ['exp1', 'exp4', 'exp5']


# In[47]:


cagemate_averages = cagemate_analysis.average_events(events = ['exp1', 'exp4', 'exp5'], mode = 'power', plot = True)


# In[98]:


novel_averages = novel_analysis.average_events(events = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5'], mode = 'power', plot = True)


# In[97]:


novel_coherence = novel_analysis.average_events(events = ['exp1', 'exp2', 'exp3','exp4', 'exp5'], mode = 'coherence', plot = True)
cagemate_coherence = cagemate_analysis.average_events(events = ['exp1', 'exp4', 'exp5'], mode = 'coherence', plot = True)


# In[141]:


importlib.reload(LFP_analysis)
cagemate_analysis = LFP_analysis.LFPAnalysis(cagemate_collection.lfp_recordings)
cagemate_analysis.plot_granger_heatmap(events, freq = [31,100])
novel_analysis = LFP_analysis.LFPAnalysis(novel_collection.lfp_recordings)
novel_analysis.plot_granger_heatmap(events, freq = [31,100])


# In[72]:


cagemate_analysis.coherence_pairs_dict


# In[79]:


cagemate_analysis.coherence_pairs_dict[frozenset({'mPFC', 'BLA'})]


# In[96]:


from scipy import stats
import matplotlib.pyplot as plt
event_averages = []
event_sems = []
recordings = []
from matplotlib.patches import Patch
# Define the events
events = ['exp1', 'exp4', 'exp5']
coherence_pair = frozenset({'NAc', 'mPFC'})
pair_index = cagemate_analysis.coherence_pairs_dict[coherence_pair]
freq_range = [12, 30]

# Loop through each recording and each event
for recording in cagemate_analysis.collection:
    rec_info = recording.recording_name + recording.subject
    recordings.append(rec_info)
    for event in events:
        event_average = cagemate_analysis.__get_event_averages__(recording,
                                                             event=event,
                                                             mode='coherence',
                                                             event_len=None,
                                                             pre_window=0,
                                                             post_window=0)
        event_array = np.array(event_average)
        event_average = np.nanmean(event_array[:, freq_range[0]:freq_range[1], pair_index[0], pair_index[1] ], axis=1)
        event_mean = np.nanmean(np.array(event_average[:]), axis=0)
        event_sem = stats.sem(np.array(event_average), axis=0, nan_policy='omit')
        event_averages.append(event_mean)
        event_sems.append(event_sem)

# Define the number of bars and their positions
num_recordings = len(cagemate_analysis.collection)
num_events = len(events)
bar_width = 0.2
x = np.arange(num_recordings)

# Define colors for each event
colors = ['skyblue', 'lightgreen', 'lightcoral']
color_map = {events[i]: colors[i] for i in range(num_events)}

# Create the bar plot
plt.figure(figsize=(10, 8))

# Plot each bar with appropriate color and error bars
for i, recording in enumerate(recordings):
    for j, event in enumerate(events):
        idx = i * num_events + j
        plt.barh(x[i] + j * bar_width, event_averages[idx], xerr=event_sems[idx],
                 capsize=5, color=color_map[event], edgecolor='black', height=bar_width,
                 label=event if i == 0 else '')

# Add labels and title
plt.ylabel('Recordings', fontsize=14)
plt.xlabel('Value', fontsize=14)
plt.title(f'Freq {freq_range} {coherence_pair} coherence', fontsize=16)

# Set custom y-ticks
ytick_positions = x + bar_width
plt.yticks(ytick_positions, recordings)

# Add a legend
handles = [Patch(color=color, label=event) for event, color in color_map.items()]
plt.legend(handles=handles, fontsize=12)

# Customize the plot (optional)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()

# Show the plot
plt.show()


# In[50]:


cagemate_analysis.brain_region_dict


# In[31]:


len(cagemate_analysis.brain_region_dict.keys())


# In[43]:


brain_regions = np.empty(5, dtype='<U10')

brain_regions[0] = 'mPFC'
brain_regions


# In[107]:


cagemate_collection.brain_region_dict = cagemate_collection.lfp_recordings[0].brain_region_dict
cagemate_collection.frequencies = cagemate_collection.lfp_recordings[0].frequencies


# In[117]:


novel_collection.brain_region_dict = novel_collection.lfp_recordings[0].brain_region_dict
novel_collection.frequencies = novel_collection.lfp_recordings[0].frequencies


# In[145]:


import lfp.lfp_analysis.LFP_analysis as lfp_tests
importlib.reload(lfp_tests)

lfp_tests.all_set(cagemate_collection)

averages = lfp_tests.plot_coherence_bar(cagemate_collection, events)


# In[ ]:





# In[150]:


import lfp.lfp_analysis.LFP_analysis as lfp_tests
importlib.reload(lfp_tests)

lfp_tests.all_set(novel_collection)

averages = lfp_tests.plot_coherence_bar(novel_collection, events = ['exp1', 'exp2', 'exp3', 'exp4', 'exp5'])


# In[246]:


for recording in cagemate_collection.lfp_recordings:
    print(recording.grangers.shape)


# In[132]:


import lfp.lfp_analysis.LFP_analysis as lfp_tests
importlib.reload(lfp_tests)
power_averages = lfp_tests.average_events(cagemate_collection, events, mode = 'power')
banded_power1, banded_power2 = lfp_tests.band_calcs(power_averages)


# In[242]:


import lfp.lfp_analysis.preprocessor as prep

for recording in cagemate_collection.lfp_recordings:
    recording.plot_to_find_threshold(threshold = 4)


# In[133]:


print(banded_power1.keys()) # events, bands
print(banded_power1['exp1'].keys())
print(banded_power1['exp1']['delta'].shape) #trials brain regions



# In[130]:


print(power_averages['exp1'][0].shape)


# In[239]:


import lfp.lfp_analysis.decoders as decoders
importlib.reload(decoders)
testing, shuffle = decoders.trial_decoder(cagemate_collection, num_fold = 5, mode = 'power', events = ['exp1', 'exp5'])

testing


# In[237]:


testing, shuffle = decoders.trial_decoder(cagemate_collection, num_fold = 5, mode = 'power', events = ['exp1', 'exp4'])

testing


# In[233]:


testing, shuffle = decoders.trial_decoder(novel_collection, num_fold = 5, mode = 'power', events = ['exp1', 'exp4'])
testing


# In[224]:


testing, shuffle = decoders.trial_decoder(novel_collection, num_fold = 5, mode = 'coherence', events = events)
testing


# In[181]:


import numpy as np
from itertools import combinations, product
def __reshape_data__(data, mode):
    stacked_bands = np.stack(data, axis=0)
    print(stacked_bands.shape)
    # stacked bands = [band, trials, regions]
    # what i want = [trials, bandxregions]
    if mode != "coherence":
        if mode == "power":
            reshaped_bands = np.transpose(stacked_bands, (1, 0, 2))
            features = list(product(range(stacked_bands.shape[0]), range(stacked_bands.shape[2])))
            reshaped_bands = reshaped_bands.reshape(reshaped_bands.shape[0], -1)
    if mode == 'coherence':
        reshaped_bands, features = __reshape_coherence_data__(stacked_bands)
    if mode == "granger":
        reshaped_bands, features = __reshape_granger_data__(stacked_bands)
    return reshaped_bands, features



def __reshape_coherence_data__(stacked_bands):
    n_bands, n_trials, n_regions, _ = stacked_bands.shape
    # Get indices for upper triangle (excluding diagonal)
    region_pairs = list(combinations(range(n_regions), 2))
    # Initialize output array
    # Shape will be [trials, bands * number_of_unique_pairs]
    n_pairs = len(region_pairs)
    reshaped = np.zeros((n_trials, n_bands * n_pairs))
    feature_indices = []
    # Fill the array
    for band in range(n_bands):
        for pair_idx, (i, j) in enumerate(region_pairs):
            # Get position in final array
            output_idx = band * n_pairs + pair_idx
            reshaped[:, output_idx] = stacked_bands[band, :, i, j]
            feature_indices.append(tuple([band, i, j]))
    return reshaped, feature_indices

def __reshape_granger_data__(stacked_bands):
    n_bands, n_trials, n_regions, _ = stacked_bands.shape
    # Get off-diagonal indices
    region_pairs = [(i, j) for i, j in product(range(n_regions), range(n_regions)) if i != j]
    print(region_pairs)
    # Initialize output array
    n_pairs = len(region_pairs)
    reshaped = np.zeros((n_trials, n_bands * n_pairs))
    feature_indices = []

    # Fill the array
    for band in range(n_bands):
        for pair_idx, (i, j) in enumerate(region_pairs):
            output_idx = band * n_pairs + pair_idx
            reshaped[:, output_idx] = stacked_bands[band, :, i, j]
            feature_indices.append(tuple([band, i, j]))

    return reshaped, feature_indices

def create_test_datasets():
    # Common parameters
    n_bands = 2
    n_regions = 3
    n_trials = 4

    # 1. Power data: shape [bands, trials, regions]
    power_data = np.array([
        # Band 1
        [[0.5, 0.7, 0.9],   # Trial 1, [region1, region2, region3]
         [0.6, 0.8, 1.0],   # Trial 2
         [0.4, 0.9, 1.1],   # Trial 3
         [0.5, 0.7, 1.2]],  # Trial 4

        # Band 2
        [[1.1, 1.3, 1.5],   # Trial 1
         [1.2, 1.4, 1.6],   # Trial 2
         [1.0, 1.5, 1.7],   # Trial 3
         [1.1, 1.3, 1.8]]   # Trial 4
    ])

    # 2. Coherence data: shape [bands, trials, regions, regions] (symmetric)
    coherence_data = np.array([
        # Band 1
        [[[1.0, 0.3, 0.4],    # Trial 1
          [0.3, 1.0, 0.5],
          [0.4, 0.5, 1.0]],
         [[1.0, 0.4, 0.5],    # Trial 2
          [0.4, 1.0, 0.6],
          [0.5, 0.6, 1.0]],
         [[1.0, 0.5, 0.6],    # Trial 3
          [0.5, 1.0, 0.7],
          [0.6, 0.7, 1.0]],
         [[1.0, 0.6, 0.7],    # Trial 4
          [0.6, 1.0, 0.8],
          [0.7, 0.8, 1.0]]],

        # Band 2
        [[[1.0, 0.6, 0.7],    # Trial 1
          [0.6, 1.0, 0.8],
          [0.7, 0.8, 1.0]],
         [[1.0, 0.7, 0.8],    # Trial 2
          [0.7, 1.0, 0.9],
          [0.8, 0.9, 1.0]],
         [[1.0, 0.8, 0.9],    # Trial 3
          [0.8, 1.0, 1.0],
          [0.9, 1.0, 1.0]],
         [[1.0, 0.9, 1.0],    # Trial 4
          [0.9, 1.0, 1.1],
          [1.0, 1.1, 1.0]]]
    ])

    # 3. Granger data: shape [bands, trials, regions, regions] (non-symmetric)
    granger_data = np.array([
        # Band 1
        [[[0.0, 0.3, 0.4],    # Trial 1
          [0.5, 0.0, 0.6],
          [0.7, 0.8, 0.0]],
         [[0.0, 0.4, 0.5],    # Trial 2
          [0.6, 0.0, 0.7],
          [0.8, 0.9, 0.0]],
         [[0.0, 0.5, 0.6],    # Trial 3
          [0.7, 0.0, 0.8],
          [0.9, 1.0, 0.0]],
         [[0.0, 0.6, 0.7],    # Trial 4
          [0.8, 0.0, 0.9],
          [1.0, 1.1, 0.0]]],

        # Band 2
        [[[0.0, 0.7, 0.8],    # Trial 1
          [0.9, 0.0, 1.0],
          [1.1, 1.2, 0.0]],
         [[0.0, 0.8, 0.9],    # Trial 2
          [1.0, 0.0, 1.1],
          [1.2, 1.3, 0.0]],
         [[0.0, 0.9, 1.0],    # Trial 3
          [1.1, 0.0, 1.2],
          [1.3, 1.4, 0.0]],
         [[0.0, 1.0, 1.1],    # Trial 4
          [1.2, 0.0, 1.3],
          [1.4, 1.5, 0.0]]]
    ])

    return power_data, coherence_data, granger_data

# Create and verify the shapes
power, coherence, granger = create_test_datasets()
print("Power shape:", power.shape)        # Should be (2, 4, 3)
print("Coherence shape:", coherence.shape) # Should be (2, 4, 3, 3)
print("Granger shape:", granger.shape)

# Create and verify the shapes
power, coherence, granger = create_test_datasets()
print("Power shape:", power.shape)        # Should be (2, 3, 2)
print("Coherence shape:", coherence.shape) # Should be (2, 3, 2, 2)
print("Granger shape:", granger.shape)    # Should be (2, 3, 2, 2)


# In[202]:





# In[183]:


print(granger) #bands, trials, regions
new_data, features = __reshape_data__(granger, mode = 'granger')
print(new_data.shape, new_data, features)


# In[249]:


for recording in cagemate_collection.lfp_recordings:
    print(np.nanmin(recording.grangers[:,40,0,3]))


# In[256]:


#TODO see if power has more or less nans than traces , does a whole window became nan if there sone nan
# find nan in trace, see if corresponding time bin in power also has nans


# In[257]:


for recording in cagemate_collection.lfp_recordings:
    #print(np.nanmax(recording.grangers[:,40,0,3]))
    x = recording.grangers[:,:,:,:]
    print('over 1', (x > 1).mean() * 100)
    print('under 1', (x < 0).mean() * 100 )
    print(recording.traces())


# In[ ]:





