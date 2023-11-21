import os
import glob
import numpy as np
import pandas as pd
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

OUTPUT_DIR = r"./proc/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

CHANNEL_MAPPING_DF = pd.read_excel("../../data/channel_mapping.xlsx")
TONE_TIMESTAMP_DF = pd.read_excel("../../data/rce_tone_timestamp.xlsx", index_col=0)

#hyper params
EPHYS_SAMPLING_RATE = 20000
LFP_SAMPLING_RATE = 1000
TRIAL_DURATION = 10
#remeber trial duration should not be a single variable but should be calculated using start and stop time input parameters
FRAME_RATE = 22
ECU_STREAM_ID = "ECU"
TRODES_STREAM_ID = "trodes"
LFP_FREQ_MIN = 0.5
LFP_FREQ_MAX = 300
ELECTRIC_NOISE_FREQ = 60
RECORDING_EXTENTION = "*.rec"


#searches for all .rec files in the data folder
ALL_SESSION_DIR = glob.glob("/scratch/back_up/reward_competition_extention/data/omission/*/*.rec")



def compute_sorted_index(group, value_column='Value', index_column='SortedIndex'):
    """
    Computes the index of each row's value within its sorted group.

    Parameters:
    - group (pd.DataFrame): A group of data.
    - value_column (str): Name of the column containing the values to be sorted.
    - index_column (str): Name of the new column that will contain the indices.

    Returns:
    - pd.DataFrame: The group with an additional column containing the indices.
    """
    #transform a column of categorical data into numerical indices based on the order of appearance in the sorted unique values
    #encoding categorical data in machine learning
    sorted_values = sorted(list(set(group[value_column].tolist())))
    group[index_column] = group[value_column].apply(lambda x: sorted_values.index(x))
    return group

#reformatting

#megans event dict = leo tone timestamp
def reformat_df(): #this whole function does not go into the generalized script
    all_trials_df = TONE_TIMESTAMP_DF.dropna(subset="condition").reset_index(drop=True)
    sorted(all_trials_df["recording_dir"].unique())

    all_trials_df["video_frame"] = all_trials_df["video_frame"].astype(int)
    all_trials_df["video_name"]  = all_trials_df["video_file"].apply(lambda x: x.strip(".videoTimeStamps.cameraHWSync"))

    # using different id extractions for different file formats
    # id file name --> look megan code
    all_trials_df["all_subjects"] = all_trials_df["recording_dir"].apply(lambda x: x if "2023" in x else "subj" + "_".join(x.split("_")[-5:]))
    all_trials_df["all_subjects"] = all_trials_df["all_subjects"].apply(lambda x: tuple(sorted([num.strip("_").replace("_",".") for num in x.replace("-", "_").split("subj")[-1].strip("_").split("and")])))
    all_trials_df["all_subjects"].unique()

    all_trials_df["current_subject"] = all_trials_df["subject_info"].apply(lambda x: ".".join(x.replace("-","_").split("_")[:2])).astype(str)
    all_trials_df["current_subject"].unique()

    #converting trial label to win or los based on which subject won trial
    all_trials_df["trial_outcome"] = all_trials_df.apply(
        lambda x: "win" if str(x["condition"]).strip() == str(x["current_subject"])
                 else ("lose" if str(x["condition"]) in x["all_subjects"]
                       else x["condition"]), axis=1)
    all_trials_df["trial_outcome"].unique()
    #TODO: what is competition closeness?
    competition_closeness_map = {k: "non_comp" if "only" in str(k).lower() else "comp" if type(k) is str else np.nan for k in all_trials_df["competition_closeness"].unique()}
    all_trials_df["competition_closeness"] = all_trials_df["competition_closeness"].map(competition_closeness_map)
    all_trials_df["competition_closeness"] = all_trials_df.apply(lambda x: "_".join([str(x["trial_outcome"]), str(x["competition_closeness"])]).strip("nan").strip("_"), axis=1)
    all_trials_df["competition_closeness"].unique()


    #STANDARDIZED STARTS HERE
    all_trials_df["lfp_index"] = (all_trials_df["time_stamp_index"] // (EPHYS_SAMPLING_RATE/LFP_SAMPLING_RATE)).astype(int)

    all_trials_df["time"] = all_trials_df["time"].astype(int)
    all_trials_df["time_stamp_index"] = all_trials_df["time_stamp_index"].astype(int)

    #ECU SPECIFIC does not need to happen
    all_trials_df = all_trials_df.drop(columns=["state", "din", "condition", "Unnamed: 13"], errors="ignore")

    #handleing time stamps
    #TODO: timestamp or frame ranges relative to LFP, ephys, and video frames.
    #To some degree all of this below should be done by the user before input 
    #but when writing doc strings , remember to clarify the units you are requesting i.e. ms
    all_trials_df["baseline_lfp_timestamp_range"] = all_trials_df["lfp_index"].apply(
        lambda x: (x - TRIAL_DURATION * LFP_SAMPLING_RATE, x))
    all_trials_df["trial_lfp_timestamp_range"] = all_trials_df["lfp_index"].apply(
        lambda x: (x, x + TRIAL_DURATION * LFP_SAMPLING_RATE))
    all_trials_df["baseline_ephys_timestamp_range"] = all_trials_df["time_stamp_index"].apply(
        lambda x: (x - TRIAL_DURATION * EPHYS_SAMPLING_RATE, x))
    all_trials_df["trial_ephys_timestamp_range"] = all_trials_df["time_stamp_index"].apply(
        lambda x: (x, x + TRIAL_DURATION * EPHYS_SAMPLING_RATE))
    all_trials_df["baseline_videoframe_range"] = all_trials_df["video_frame"].apply(
        lambda x: (x - TRIAL_DURATION * FRAME_RATE, x))
    all_trials_df["trial_videoframe_range"] = all_trials_df["video_frame"].apply(
        lambda x: (x, x + TRIAL_DURATION * FRAME_RATE))
    return all_trials_df


def extract_lfp(): #generlized script starts here
    # Going through all the recording sessions
    recording_name_to_all_ch_lfp = {}
    for session_dir in ALL_SESSION_DIR:
        # Going through all the recordings in each session
        for recording_path in glob.glob(os.path.join(session_dir, RECORDING_EXTENTION)):
            #assumes subject name in file name
            try:
                recording_basename = os.path.splitext(os.path.basename(recording_path))[0]
                # checking to see if the recording has an ECU component
                # if it doesn't, then the next one be extracted
                current_recording = se.read_spikegadgets(recording_path, stream_id=ECU_STREAM_ID)
                current_recording = se.read_spikegadgets(recording_path, stream_id=TRODES_STREAM_ID) #we need to confer with leo what these lines do
                print(recording_basename)
                # Preprocessing the LFP
                # higher than 300 is action potential and lower than 0.5 is noise
                current_recording = sp.bandpass_filter(current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
                current_recording = sp.notch_filter(current_recording, freq=ELECTRIC_NOISE_FREQ)
                current_recording = sp.resample(current_recording, resample_rate=LFP_SAMPLING_RATE)
                # Z-scoring the LFP
                current_recording = sp.zscore(current_recording) # zscore single because avg across animals is in same scale
                recording_name_to_all_ch_lfp[recording_basename] = current_recording
            except Exception as error:
                # handle the exception
                print("An exception occurred:", error)  # An exception occurred: division by zero
    return recording_name_to_all_ch_lfp

all_trials_df = reformat_df()

recording_name_to_all_ch_lfp = extract_lfp()

all_trials_df = all_trials_df[all_trials_df["recording_file"].isin(recording_name_to_all_ch_lfp.keys())].reset_index(drop=True)
all_trials_df = all_trials_df.groupby('recording_file').apply(lambda g: compute_sorted_index(g, value_column='time', index_column='trial_number')).reset_index(drop=True)
all_trials_df["trial_number"] = all_trials_df["trial_number"] + 1

#adding the LFP trace information
#todo: make function
CHANNEL_MAPPING_DF["Subject"] = CHANNEL_MAPPING_DF["Subject"].astype(str)
channel_map_and_all_trials_df = all_trials_df.merge(CHANNEL_MAPPING_DF, left_on="current_subject", right_on="Subject", how="left")
channel_map_and_all_trials_df = channel_map_and_all_trials_df.drop(columns=[col for col in channel_map_and_all_trials_df.columns if "eib" in col], errors="ignore")
channel_map_and_all_trials_df = channel_map_and_all_trials_df.drop(columns=["Subject"], errors="ignore")
channel_map_and_all_trials_df.to_csv("./proc/trial_metadata.csv")
channel_map_and_all_trials_df.to_pickle("./proc/trial_metadata.pkl")

#subsampling and channel mapping would be functions of ephys coillection
#link lfp to trials
channel_map_and_all_trials_df["all_ch_lfp"] = channel_map_and_all_trials_df["recording_file"].map(recording_name_to_all_ch_lfp)
#new row for brain region
brain_region_col = [col for col in CHANNEL_MAPPING_DF if "spike_interface" in col]
id_cols = [col for col in channel_map_and_all_trials_df.columns if col not in brain_region_col]
for col in brain_region_col:
    # object stuff for ecu specific
    channel_map_and_all_trials_df[col] = channel_map_and_all_trials_df[col].astype(int).astype(str)
    channel_map_and_all_trials_df["{}_baseline_lfp_trace".format(col.strip("spike_interface").strip("_"))] = channel_map_and_all_trials_df.apply(lambda row: row["all_ch_lfp"].get_traces(channel_ids=[row[col]], start_frame=row["baseline_lfp_timestamp_range"][0], end_frame=row["baseline_lfp_timestamp_range"][1]).T[0], axis=1)
    channel_map_and_all_trials_df["{}_trial_lfp_trace".format(col.strip("spike_interface").strip("_"))] = channel_map_and_all_trials_df.apply(lambda row: row["all_ch_lfp"].get_traces(channel_ids=[row[col]], start_frame=row["trial_lfp_timestamp_range"][0], end_frame=row["trial_lfp_timestamp_range"][1]).T[0], axis=1)
channel_map_and_all_trials_df = channel_map_and_all_trials_df.drop(columns=["all_ch_lfp"], errors="ignore")
channel_map_and_all_trials_df.to_pickle("./proc/full_baseline_and_trial_lfp_traces.pkl")


# spectogram -- power over time and freq --heatmap -- spectral_connectivity
    # sliding window
# pow correlation -- power on windows -- two brain regions -- correlation (up or down power together)
    # pow in 1 sec windows across tone times
    # pow across interactions  -- scatter plot
    # window size = 1 - 2 seconds
    # really small window size -- gamma (more time res on gamma)

# power -- amplitude across freq -- spectral_connectivity
# coherence -- phase consistency of two regions
    # phase = hill vs trough
    # "how much they line up over time"
    # func tells how much of signal is predictive of other signal
    # coherence = 1 -- signals are perfectly lined up
    # coherence = 0.5 -- signals are not lined up at all
    # coherence = 0 -- signals are perfectly out of phase
    # coherence = 1 - 0.5 or 0.5 - 0 -- signals are in phase
    # time lag -- how much time is between signals

class LFPrecordingCollection:
    def __init__(self,
                 path,
                 tone_timestamp_df, # this should be a dict or simplified df
                 channel_mapping_df,
                 all_sessions_path,
                 ephys_sampling_rate,
                 lfp_sampling_rate,
                 trial_duration, #this should be baked into the dict above i.e. tone_timestamp_df 
                 frame_rate=22, #we should figure out if this is actually needed
                 pickle_path="./proc/full_baseline_and_trial_lfp_traces.pkl", #let this be its own function with user input
                 recording_extention="*.rec", # i dont think this needs to be a variable just hard code this one 
                 lfp_freq_min=0.5,
                 lfp_freq_max=300,
                 electric_noise_freq=60,
                 ecu_stream_id="ECU",
                 trodes_stream_id = "trodes"):

        #make ecu default is false and if true go to extract function
        self.ephys_sampling_rate = ephys_sampling_rate
        self.lfp_sampling_rate = lfp_sampling_rate
        self.frame_rate = frame_rate
        self.trial_duration = trial_duration #GET RID OF THIS--> start & stop col
        self.recording_extention = recording_extention
        self.path = path # i htink this is synonymous with all_sessions_path, pick path its simpler 
        self.lfp_freq_min = lfp_freq_min
        self.lfp_freq_max = lfp_freq_max
        self.electric_noise_freq = electric_noise_freq
        self.ecu_stream_id = ecu_stream_id
        self.trodes_stream_id = trodes_stream_id

        #change to path

        # later object
        self.tone_timestamp_df = pd.read_excel(tone_timestamp_df, index_col=0) #this line will be removed - the input will not be an excel path but either a behavior df or dict 

        # path to excel, convert to df --> property of obj
        self.channel_mapping_df = pd.read_excel(channel_mapping_df)

        self.all_trials_df = self.reformat_df() #this is a leo specific thing get rid of this 

        self.recording_name_to_all_ch_lfp = self.extract_lfp() #this is a terrible attribute name, lets talk about what this is actaully creating 
        # replace with just path
        self.all_sessions_dir = glob.glob(all_sessions_path) #if youve saved all_sessions path as path you dont need to save this as an attribute as well

        self.channel_map_and_all_trials_df = self.add_lfp_trace()


        #do not need pickle path in the init

        self.pickle = self.channel_map_and_all_trials_df.to_pickle(pickle_path)


    #TODO: get rid of or put in second object, any time dealing with tone timestamp put behvior
    #can be a function within the object but cannot be called in init function
    #make all trials df compatible with collection object from spike

    #ecu extraction for data analysis

    #helper

    def reformat_df(self): # i dont want this is i nteh class= remove it outside of the class
        #behavior stuff
        all_trials_df = self.tone_timestamp_df.dropna(subset="condition").reset_index(drop=True)
        sorted(all_trials_df["recording_dir"].unique())

        all_trials_df["video_frame"] = all_trials_df["video_frame"].astype(int)
        all_trials_df["video_name"] = all_trials_df["video_file"].apply(
            lambda x: x.strip(".videoTimeStamps.cameraHWSync"))

        # using different id extractions for different file formats
        # id file name --> look megan code
        all_trials_df["all_subjects"] = all_trials_df["recording_dir"].apply(
            lambda x: x if "2023" in x else "subj" + "_".join(x.split("_")[-5:]))
        all_trials_df["all_subjects"] = all_trials_df["all_subjects"].apply(lambda x: tuple(sorted(
            [num.strip("_").replace("_", ".") for num in
             x.replace("-", "_").split("subj")[-1].strip("_").split("and")])))
        all_trials_df["all_subjects"].unique()

        all_trials_df["current_subject"] = all_trials_df["subject_info"].apply(
            lambda x: ".".join(x.replace("-", "_").split("_")[:2])).astype(str)
        all_trials_df["current_subject"].unique()

        # converting trial label to win or los based on which subject won trial
        all_trials_df["trial_outcome"] = all_trials_df.apply(
            lambda x: "win" if str(x["condition"]).strip() == str(x["current_subject"])
            else ("lose" if str(x["condition"]) in x["all_subjects"]
                  else x["condition"]), axis=1)
        all_trials_df["trial_outcome"].unique()

        # TODO: specific to this expr, not general
        # no function should assume inputs have win or lose, iterate through
        # one dict with n events as keys --> values of this start and stop times

        competition_closeness_map = {k: "non_comp" if "only" in str(k).lower() else "comp" if type(k) is str else np.nan
                                     for k in all_trials_df["competition_closeness"].unique()}
        all_trials_df["competition_closeness"] = all_trials_df["competition_closeness"].map(competition_closeness_map)
        all_trials_df["competition_closeness"] = all_trials_df.apply(
            lambda x: "_".join([str(x["trial_outcome"]), str(x["competition_closeness"])]).strip("nan").strip("_"),
            axis=1)
        all_trials_df["competition_closeness"].unique()

        # STANDARDIZED STARTS HERE
        all_trials_df["lfp_index"] = (
                    all_trials_df["time_stamp_index"] // (self.ephys_sampling_rate / self.lfp_sampling_rate)).astype(int)

        all_trials_df["time"] = all_trials_df["time"].astype(int)
        all_trials_df["time_stamp_index"] = all_trials_df["time_stamp_index"].astype(int)

        # ECU SPECIFIC does not need to happen
        all_trials_df = all_trials_df.drop(columns=["state", "din", "condition", "Unnamed: 13"], errors="ignore")

        # handleing time stamps
        # TODO: timestamp or frame ranges relative to LFP, ephys, and video frames.
        all_trials_df["baseline_lfp_timestamp_range"] = all_trials_df["lfp_index"].apply(
            lambda x: (x - self.trial_duration * self.lfp_sampling_rate, x))
        all_trials_df["trial_lfp_timestamp_range"] = all_trials_df["lfp_index"].apply(
            lambda x: (x, x + self.trial_duration * self.lfp_sampling_rate))
        all_trials_df["baseline_ephys_timestamp_range"] = all_trials_df["time_stamp_index"].apply(
            lambda x: (x - self.trial_duration * self.ephys_sampling_rate, x))
        all_trials_df["trial_ephys_timestamp_range"] = all_trials_df["time_stamp_index"].apply(
            lambda x: (x, x + self.trial_duration * self.ephys_sampling_rate))
        all_trials_df["baseline_videoframe_range"] = all_trials_df["video_frame"].apply(
            lambda x: (x - self.trial_duration * self.frame_rate, x))
        all_trials_df["trial_videoframe_range"] = all_trials_df["video_frame"].apply(
            lambda x: (x, x + self.trial_duration * self.frame_rate))
        return all_trials_df

    #megan reading in numpy array
    def extract_lfp(self):
        # Going through all the recording sessions
        recording_name_to_all_ch_lfp = {}
        for session_dir in self.all_sessions_dir:
            # Going through all the recordings in each session
            for recording_path in glob.glob(os.path.join(session_dir, self.recording_extention)):
                # assumes subject name in file name
                try:
                    #TODO: assumes subject name in file name
                    recording_basename = os.path.splitext(os.path.basename(recording_path))[0]
                    # checking to see if the recording has an ECU component
                    # if it doesn't, then the next one be extracted

                    #TODO: read_spikegadgets
                    #TODO: GET RID OF FIRST LINE
                    #current_recording = se.read_spikegadgets(recording_path, stream_id=self.ecu_stream_id)
                    current_recording = se.read_spikegadgets(recording_path, stream_id=self.trodes_stream_id)
                    print(recording_basename)
                    # Preprocessing the LFP
                    # higher than 300 is action potential and lower than 0.5 is noise
                    current_recording = sp.bandpass_filter(current_recording, freq_min=self.lfp_freq_min,
                                                           freq_max=self.lfp_freq_max)
                    current_recording = sp.notch_filter(current_recording, freq=self.electric_noise_freq)
                    current_recording = sp.resample(current_recording, resample_rate=self.lfp_sampling_rate)
                    # Z-scoring the LFP
                    current_recording = sp.zscore(
                        current_recording)  # zscore single because avg across animals is in same scale
                    recording_name_to_all_ch_lfp[recording_basename] = current_recording
                except Exception as error:
                    # handle the exception
                    print("An exception occurred:", error)  # An exception occurred: division by zero
        return recording_name_to_all_ch_lfp

    def add_lfp_trace(self): #please investigate what each line does, i feel like we do not need to this many things but rather just create a new 
        #attribute instead of manipulating a df in these very precise ways , stop pruning and only take what you need
        self.channel_mapping_df["Subject"] = self.channel_mapping_df["Subject"].astype(str)
        channel_map_and_all_trials_df = all_trials_df.merge(self.channel_mapping_df, left_on="current_subject",
                                                            right_on="Subject", how="left") 
        channel_map_and_all_trials_df = channel_map_and_all_trials_df.drop(
            columns=[col for col in channel_map_and_all_trials_df.columns if "eib" in col], errors="ignore")
        channel_map_and_all_trials_df = channel_map_and_all_trials_df.drop(columns=["Subject"], errors="ignore")
        channel_map_and_all_trials_df.to_csv("./proc/trial_metadata.csv")
        channel_map_and_all_trials_df.to_pickle("./proc/trial_metadata.pkl")

        # subsampling and channel mapping would be functions of ephys coillection
        # link lfp to trials
        channel_map_and_all_trials_df["all_ch_lfp"] = channel_map_and_all_trials_df["recording_file"].map(
            recording_name_to_all_ch_lfp)
        # new row for brain region
        brain_region_col = [col for col in self.channel_mapping_df if "spike_interface" in col]
        id_cols = [col for col in channel_map_and_all_trials_df.columns if col not in brain_region_col]
        for col in brain_region_col:
            # object stuff for ecu specific
            channel_map_and_all_trials_df[col] = channel_map_and_all_trials_df[col].astype(int).astype(str)
            channel_map_and_all_trials_df["{}_baseline_lfp_trace".format(
                col.strip("spike_interface").strip("_"))] = channel_map_and_all_trials_df.apply(lambda row: row[
                "all_ch_lfp"].get_traces(channel_ids=[row[col]], start_frame=row["baseline_lfp_timestamp_range"][0],
                                         end_frame=row["baseline_lfp_timestamp_range"][1]).T[0], axis=1)
            channel_map_and_all_trials_df["{}_trial_lfp_trace".format(
                col.strip("spike_interface").strip("_"))] = channel_map_and_all_trials_df.apply(lambda row: row[
                "all_ch_lfp"].get_traces(channel_ids=[row[col]], start_frame=row["trial_lfp_timestamp_range"][0],
                                         end_frame=row["trial_lfp_timestamp_range"][1]).T[0], axis=1)
        channel_map_and_all_trials_df = channel_map_and_all_trials_df.drop(columns=["all_ch_lfp"], errors="ignore")
        return channel_map_and_all_trials_df


    #rewrite {event: [start, stop]} --> {event: [start, stop, start, stop]}
        #recording, subject
        #take all trials and separate into individual dfs per recording 
    #reformat function to use above format
    #add print statements to exception


    #remove reformat function outside of class
    #follow up function
        #input --> all_trials_df
        #output --> rewritten all_trials_df
    #separate function for assigning (current) subject
        #assign to each recording a current subject
        #input --> all_trials_df
    #rewrite add_lfp_trace and extract to use newlly formatted all_trials_df


