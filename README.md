# diff_fam_social_memory_ephys
This repository holds code to conduct ephys analysis on data files that come from trodes and phy files. It also has code to extract behavior data from Boris and ECU data and some tools to manipulate behavioral epochs. 

### Prerequisites
- conda or miniconda installed

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/padillacoreanolab/diff_fam_social_memory_ephys.git
   ```
   or use github desktop to clone it locally. 
   
   Open the repository in a terminal that can access conda


2. Change the working directory on VSCode to this diff_fam_social_memory_ephys folder itself. Go to the settings tab under workspace and change the following setting: 
- (File>Preference>Settings or Ctrl+,)

- Jupyter: Notebook File Root
-- ${workspaceFolder}

this means that the parent folder opened in VSCode is the workspaceFolder which is set as the working directory which if you opened the whole folder, diff_fam_social_memory_ephys, this should be correct. 

**note: where a jupyternotebook lives is not necessarily where
it is being run, those are two different things. 

- This is important because it is brittle to have imports be relative to where the jupyter notebook file is stored for importing local modules and extremely important for [pickling and unpickling](https://stackoverflow.com/a/2121918). Eg, if you wanted to move notebooks, or have notebooks be portable from computer to computer[text]


3. Create and activate conda environment
```
conda env create -f ephys_env.yml
conda activate ephys_env
```
4. For house-keeping, create your own folder under other_peoples_stuff/YOURNAME. 

5. If you set up the working directory setting correctly you should be able to import functions like this:
```
from lfp.lfp_analysis.lfp_collection import LFPCollection
from spike.spike_analysis.spike_collection import SpikeCollection
```

## What do you need
This whole code base is made under the assumption you have left the automatic naming conventions of trodes and phy. Meaning, this code will not work if: for trodes files 1.) you have not left .rec at the end of every recording folder or 2.) you not kept the merged.rec ending for every recording file and for phy outputs you need 1.) a folder ending with merged.rec and inside that folder is a 2.) phy folder 
- LFP: Current version of this code assumes you have recorded your data through Trodes and have kept the *.rec/*merged.rec data folder structure. 
- Timestamps: you will need the timestamps.dat folder in a .time folder
- Spike: current version of this code assumes you have done manual spike sorting through Phy and have saved everything under a *merged.rec/phy/.. data structure. 
- Behavior: both Spike and LFP analysis require the same format of behavior input for analysis, currently helper functions exist under Behavior for BORIS and ECU derived behavior data. 

## Odds and ends of getting all the things you need

### Extracting relevant files: LFP + ECU + Analog 
trodes_export.py has functions that will allow you batch export various files through trodes. It is under the trodes folder. You will need trodes installed onto your local device to use its functions. 
- time: export needed to find the first timestamp of a recording for all LFP, ECU and analog analysis
- DIO: if you used the ECU to record operant chamber inputs, you will need to extract DIO files 
- analog: if you used any Ain inputs to record, you will need to extract analog files 

### Converting h264 videos into sleap friendly mp4s
Under helper functions is also a suite of conversion functions to convert h264 files (or any video format) into a sleap friendly mp4. This also helps if you want to use BORIS or AnyMaze or any other software to do behavior classification. 

# Contents of this repository 
It is highly recommended that users understand what a class is in python to most effectively use the following scripts. The lfp and spike folders each have the own README for more detailed overviews of the contents of those folders. 

## Spike 
This folder contains scripts to analyze the output of phy manually curated spikes. It contains scripts to do single cell analyses (Wilcoxon tests, raster plots, and bootstrap firing rate comparisons), PCA analyses and trajectory visualizations, classifiers to predict  behavioral events based on population activity, and functions to normalize firing rates. 

## LFP
This folder contains scripts to analyze the output of a trodes recording. It contains scripts to calculate power, coherence and Granger causality using the package [Multitaper Spectral Connectivity](https://spectral-connectivity.readthedocs.io/en/latest/index.html). It is highly recommended to use the Hipergator (or other high computing cluster) to do so. GPU greatly increases speed of computation for Granger causality specifically and allows you to more readily load all your recordings into working memory to perform data analysis and plotting.  

Below are a few instructions on how to use rclone to get your large datasets from dropbox to the remote computing cluster: 

### dropbox + rclone
1. install rclone on your local computer: https://rclone.org/downloads/
2. run on personal computer 
```
rclone.exe authorize "dropbox"
```

crtl + click on link that terminal spits out and press agree, terminal will now spit out a token key, finish step 3 before copying and pasting it into the hipergator terminal

3. open terminal in hipergator and run the following lines: 
```
module load rclone
rclone config
```

- n) New remote
- name your dropbox
- Storage > 13 (type 13 (which is dropobx) and hit enter) 
- leave client_id and client_secret blank (aka press enter) 
- type n and click enter (No) for edit advanced config
- type n and click enter (No) for web broswer question
- copy and paste config token from local terminal into hipergator terminal
- click y and hit enter for default settings to finish

if authentication breaks press choose: 
- e) Edit existing remote 
- 4. find data on dropbox
```
rclone ls pc-dropbox:"Padilla-Coreano Lab/path/to/data"
```
5. copy data from dropdox
source = pc-dropbox
path ="path/to/folder"
run dry run first to confirm path and size of download 

example of destination path
dest:path = ./data 
```
rclone copy source:path dest:path --progress --dry-run
```
then run it for reals
```
rclone copy source:path dest:path --progress
```

example real command:
```
rclone copy pc-dropbox:"Padilla-Coreano Lab/2024/Cum_SocialMemEphys_pilot2/Only_Subjects (phase 5)/sorted" ./only_subjects_spike_data --progress  --dry-run
```

or to upload data to dropbox: 

```
rclone copy ./r4_predictions pc-dropbox:"Padilla-Coreano Lab/2024/Cum_SocialMemEphys_pilot2/SLEAP/only_subject_mp4s/r4_predictions" --progress --dry-run
```
# Developer installation

## How to run tests

### Download test data

We're going to use this small recording from trodes:
https://docs.spikegadgets.com/en/latest/basic/BasicUsage.html

There is a helper script that downloads it and unzips it into `tests/test_data/`.

```
python -m tests.utils download_test_data
```

please open terminal and run in this directory

```
python -m unittest discover -s *folder_you_want_to_test* -p '*test*.py
```

## How to run single files as modules

For example, to turn a dataset into a test:

```
python -m tests.utils create /Volumes/SheHulk/cups/data/11_cups_p4.rec/11_cups_p4_merged.rec
```

This is nice because it allows all imports to be relative to the root directory.