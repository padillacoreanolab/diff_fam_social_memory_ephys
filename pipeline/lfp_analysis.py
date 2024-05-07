import glob
import subprocess
import os
from collections import defaultdict
import trodes.read_exported
import pandas as pd
import numpy as np
from scipy import stats
from spectral_connectivity import Multitaper, Connectivity
import openpyxl
import logging


import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp


# TODO: need to make collection object
#  user still needs to call the convert to mp4 function
#  need to fix pkl save path for object
#  need to more modular, power functions (from notebook 2) are automatically called at object creation
#  power, phase, coherence, granger functions depend on each other (add none exceptions)
#  need to make all df columns lower case

class LFPObject:
    def make_object(self):
        # call extract_all_trodes
        session_to_trodes_temp, paths = extract_all_trodes(self.path)
        # call add_video_timestamps
        session_to_trodes_temp = add_video_timestamps(session_to_trodes_temp, self.path)
        # call create_metadata_df
        metadata = create_metadata_df(session_to_trodes_temp, paths)
        # call adjust_first_timestamps
        metadata, state_df, video_df, final_df, pkl_path = adjust_first_timestamps(metadata, self.path, self.subject)
        # assign variables
        self.metadata = metadata
        self.state_df = state_df
        self.video_df = video_df
        self.final_df = final_df
        self.pkl_path = pkl_path


    def make_power_df(self):
        # handle modified z score
        if self.pkl_path is not None:
            print("CALLED here")
            LFP_TRACES_DF = preprocess_lfp_data(pd.read_pickle(self.pkl_path), self.VOLTAGE_SCALING_VALUE, self.zscore_threshold, self.RESAMPLE_RATE)
            self.LFP_TRACES_DF = LFP_TRACES_DF
            print("LFP TRACES DF")
            print(LFP_TRACES_DF.head())
        else:
            print("NO PKL PATH")
            return
        # call get_power
        power_df = calculate_power(self.spike_df, self.RESAMPLE_RATE, self.TIME_HALFBANDWIDTH_PRODUCT, self.TIME_WINDOW_DURATION, self.TIME_WINDOW_STEP)
        # assign variables
        self.power_df = power_df

    def make_phase_df(self):
        # call get_phase
        phase_df = calculate_phase(self.spike_df, fs=1000)
        # assign variables
        self.phase_df = phase_df

    def make_coherence_df(self):
        # call get_coherence
        #lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step
        coherence_df = calculate_coherence(self.spike_df, self.RESAMPLE_RATE, self.TIME_HALFBANDWIDTH_PRODUCT, self.TIME_WINDOW_DURATION, self.TIME_WINDOW_STEP)
        # assign variables
        self.coherence_df = coherence_df

    def make_granger_df(self):
        # call get_granger
        #lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step
        granger_df = calculate_granger_causality(lfp_traces_df=self.spike_df, resample_rate=self.RESAMPLE_RATE, time_halfbandwidth_product=self.TIME_HALFBANDWIDTH_PRODUCT, time_window_duration=self.TIME_WINDOW_DURATION, time_window_step=self.TIME_WINDOW_STEP)
        # assign variables
        self.granger_df = granger_df

    def __init__(self,
                 path,
                 channel_map_path,
                 events_path,
                 subject,
                 ecu=False,
                 sampling_rate=20000,
                 frame_rate=22):
        self.path = path
        self.channel_map_path = channel_map_path
        self.events_path = events_path
        self.events = {}
        self.channel_map = {}
        self.recording = None
        self.subject = subject
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate

        #add variables from make object function
        self.metadata = None
        self.state_df = None
        self.video_df = None
        self.final_df = None
        self.pkl_path = None

        #inputs needed for notebook 2 (power)
        self.LFP_TRACES_DF = None
        self.original_trace_columns = None

        #hard coding these variables for power
        self.VOLTAGE_SCALING_VALUE = 0.195
        self.zscore_threshold = 4
        self.RESAMPLE_RATE = 1000
        self.TIME_HALFBANDWIDTH_PRODUCT = 2
        self.TIME_WINDOW_DURATION = 1
        self.TIME_WINDOW_STEP = 0.5
        self.BAND_TO_FREQ = {"theta": (4, 12), "gamma": (30, 51)}

        #power stuff
        self.power_df = None
        self.phase_df = None
        self.coherence_df = None
        self.granger_df = None


        self.make_object()


        #get channel map and lfp
        #ALL_SESSION_DIR, ECU_STREAM_ID, TRODES_STREAM_ID, RECORDING_EXTENTION, LFP_FREQ_MIN, LFP_FREQ_MAX, ELECTRIC_NOISE_FREQ, LFP_SAMPLING_RATE, EPHYS_SAMPLING_RATE):
        self.recording_names_dict = extract_lfp_traces(ALL_SESSION_DIR=self.path, ECU_STREAM_ID="ECU", TRODES_STREAM_ID="trodes", RECORDING_EXTENTION="*.rec", LFP_FREQ_MIN=0.5, LFP_FREQ_MAX=300, ELECTRIC_NOISE_FREQ=60, LFP_SAMPLING_RATE=1000, EPHYS_SAMPLING_RATE=20000)
        self.channel_map, self.spike_df = load_data(channel_map_path=self.channel_map_path, pickle_path=self.pkl_path)
        self.spike_df = combine_lfp_traces_and_metadata(SPIKEGADGETS_EXTRACTED_DF=self.spike_df, recording_name_to_all_ch_lfp=self.recording_names_dict, CHANNEL_MAPPING_DF=self.channel_map, CURRENT_SUBJECT_COL="current_subject", SUBJECT_COL="Subject", ALL_CH_LFP_COL="all_ch_lfp", LFP_RESAMPLE_RATIO=20, EPHYS_SAMPLING_RATE=20000, LFP_SAMPLING_RATE=1000)

        #temporarily pickle the spike_df for debugging
        self.spike_df.to_pickle(os.getcwd() + "test_outputs/spike_df.pkl")

        self.make_power_df()
        self.make_phase_df()
        self.make_coherence_df()
        self.make_granger_df()



def find_nearest_indices(array1, array2):
    """
    Finds the indices of the elements in array2 that are nearest to the elements in array1.

    This function flattens array1 and for each number in the flattened array, finds the index of the
    number in array2 that is nearest to it. The indices are then reshaped to match the shape of array1.

    Parameters:
    - array1 (numpy.ndarray): The array to find the nearest numbers to.
    - array2 (numpy.ndarray): The array to find the nearest numbers in.

    Returns:
    - numpy.ndarray: An array of the same shape as array1, containing the indices of the nearest numbers
                     in array2 to the numbers in array1.
    """
    array1_flat = array1.flatten()
    indices = np.array([np.abs(array2 - num).argmin() for num in array1_flat])
    return indices.reshape(array1.shape)

def convert_to_mp4(experiment_dir):
    """
    Converts .h264 files to .mp4 files using the bash script convert_to_mp4.sh
    convert_to_mp4.sh should exist in the same directory as this script.
    Args:
        experiment_dir (String): Path to the experiment directory containing subdirectories with .h264 files.
            For example, if your experiment contains the following subdirectories:
                /path/to/experiment/trial1
                /path/to/experiment/trial2
            Your experiment_dir should be /path/to/experiment.
    Returns:
        None
    """
    bash_path = "./convert_to_mp4.sh"
    subprocess.run([bash_path, experiment_dir])

experiment_dir = "/Volumes/chaitra/test_lfp"
#convert_to_mp4(experiment_dir)
def extract_all_trodes(input_dir):
    """
    Args:
        input_dir (String): Path containing the session directories to process.

    Returns:
        session_to_trodes_data (defaultdict): A nested dictionary containing the metadata for each session.
    """

    def recursive_dict():
        return defaultdict(recursive_dict)

    session_to_trodes_data = recursive_dict()
    session_to_path = {}

    # This loop will process each session directory using the trodes extract functions and store the metadata in a
    # nested dictionary.

    for session in glob.glob(input_dir):
        try:
            session_basename = os.path.splitext(os.path.basename(session))[0]
            print("Processing session: ", session_basename)
            session_to_trodes_data[session_basename] = trodes.read_exported.organize_all_trodes_export(session) #
            session_to_path[session_basename] = session
        except Exception as e:
            print("Error processing session: ", session_basename)
            print(e)

    # print(session_to_trodes_data)
    return session_to_trodes_data, session_to_path

def add_video_timestamps(session_to_trodes_data, directory_path):
    """
    Args:
        session_to_trodes_data (Nested Dictionary): Generate from extract_all_trodes.
        directory_path (String): Path containing the session directories to process.

    Returns:
        session_to_trodes_data (Nested Dictionary): A nested dictionary containing the metadata for each session.
    """

    # Loops through each session and video_timestamps file and adds the timestamps to the session_to_trodes_data
    # dictionary. Timestamp array is generated using the read_trodes_extracted_data_file function from the
    # trodes.read_exported module.

    for session in glob.glob(directory_path):
        try:
            session_basename = os.path.splitext(os.path.basename(session))[0]
            print("Current Session: {}".format(session_basename))

            for video_timestamps in glob.glob(os.path.join(session, "*cameraHWSync")):
                video_basename = os.path.basename(video_timestamps)
                print("Current Video Name: {}".format(video_basename))
                timestamp_array = trodes.read_exported.read_trodes_extracted_data_file(video_timestamps)

                if "video_timestamps" not in session_to_trodes_data[session_basename][session_basename]:
                    session_to_trodes_data[session_basename][session_basename]["video_timestamps"] = defaultdict(dict)

                session_to_trodes_data[session_basename][session_basename]["video_timestamps"][video_basename.split(".")[-3]] = timestamp_array
                print("Timestamp Array for {}: ".format(video_basename))
                print(session_to_trodes_data[session_basename][session_basename]["video_timestamps"][video_basename.split(".")[-3]])

        except Exception as e:
            print("Error processing session: ", session_basename)
            print(e)

        return session_to_trodes_data

def create_metadata_df(session_to_trodes, session_to_path):
    """

    Args:
        session_to_trodes (nested dictionary): Generated from extract_all_trodes.
        session_to_path (empty dictionary): {}
        columns_to_keep (dictionary): Provide a dictionary of the columns to keep in the metadata dataframe.

    Returns:
        trodes_metadata_df (pandas dataframe): A dataframe containing the metadata for each session.
    """

    trodes_metadata_df = pd.DataFrame.from_dict({(i,j,k,l): session_to_trodes[i][j][k][l]
                            for i in session_to_trodes.keys()
                            for j in session_to_trodes[i].keys()
                            for k in session_to_trodes[i][j].keys()
                            for l in session_to_trodes[i][j][k].keys()},
                            orient='index')

    trodes_metadata_df = trodes_metadata_df.reset_index()
    trodes_metadata_df = trodes_metadata_df.rename(columns={'level_0': 'session_dir', 'level_1': 'recording', 'level_2': 'metadata_dir', 'level_3': 'metadata_file'}, errors="ignore")
    trodes_metadata_df["session_path"] = trodes_metadata_df["session_dir"].map(session_to_path)

    # Adjust data types
    trodes_metadata_df["first_dtype_name"] = trodes_metadata_df["data"].apply(lambda x: x.dtype.names[0])
    trodes_metadata_df["first_item_data"] = trodes_metadata_df["data"].apply(lambda x: x[x.dtype.names[0]])
    trodes_metadata_df["last_dtype_name"] = trodes_metadata_df["data"].apply(lambda x: x.dtype.names[-1])
    trodes_metadata_df["last_item_data"] = trodes_metadata_df["data"].apply(lambda x: x[x.dtype.names[-1]])

    print("unique recordings ")
    print(trodes_metadata_df["recording"].unique())
    return trodes_metadata_df

def add_subjects_to_metadata(metadata):
    # TODO: find a better way to do this without regex on the session_dir
    metadata["all_subjects"] = metadata["session_dir"].apply(
        lambda x: x.replace("-", "_").split("subj")[-1].split("t")[0].strip("_").replace("_", ".").split(".and."))
    metadata["all_subjects"] = metadata["all_subjects"].apply(
        lambda x: sorted([i.strip().strip(".") for i in x]))
    metadata["current_subject"] = metadata["recording"].apply(
        lambda x: x.replace("-", "_").split("subj")[-1].split("t")[0].strip("_").replace("_", ".").split(".and.")[0])
    print(metadata["all_subjects"])
    print(metadata["current_subject"])
    return metadata

def get_trodes_video_df(trodes_metadata_df):
    trodes_video_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"] == "video_timestamps"].copy().reset_index(
        drop=True)
    trodes_video_df = trodes_video_df[trodes_video_df["metadata_file"] == "1"].copy()
    trodes_video_df["video_timestamps"] = trodes_video_df["first_item_data"]
    trodes_video_df = trodes_video_df[["filename", "video_timestamps", "session_dir"]].copy()
    trodes_video_df = trodes_video_df.rename(columns={"filename": "video_name"})
    print(trodes_video_df.head())
    return trodes_video_df

def get_trodes_state_df(trodes_metadata_df):
    trodes_state_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"].isin(["DIO"])].copy()
    trodes_state_df = trodes_metadata_df[trodes_metadata_df["id"].isin(["ECU_Din1", "ECU_Din2"])].copy()
    trodes_state_df["event_indexes"] = trodes_state_df.apply(
        lambda x: np.column_stack([np.where(x["last_item_data"] == 1)[0], np.where(x["last_item_data"] == 1)[0] + 1]),
        axis=1)
    trodes_state_df["event_indexes"] = trodes_state_df.apply(
        lambda x: x["event_indexes"][x["event_indexes"][:, 1] <= x["first_item_data"].shape[0] - 1], axis=1)
    trodes_state_df["event_timestamps"] = trodes_state_df.apply(lambda x: x["first_item_data"][x["event_indexes"]],
                                                                axis=1)
    print(trodes_state_df.head())
    return trodes_state_df

def get_trodes_raw_df(trodes_metadata_df):
    """
    Extracts the raw data from the trodes_metadata_df and calculates the first timestamp for each session.
    Args:
        trodes_metadata_df (pandas dataframe): Generated from create_metadata_df.
    Returns:
        trodes_raw_df (pandas dataframe): A dataframe containing the raw data for each session.

    """
    trodes_raw_df = trodes_metadata_df[
        (trodes_metadata_df["metadata_dir"] == "raw") & (trodes_metadata_df["metadata_file"] == "timestamps")].copy()
    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(lambda x: x[0])
    trodes_raw_cols = ['session_dir', 'recording', 'original_file', 'session_path', 'current_subject', 'first_item_data',
                       'first_timestamp','all_subjects']
    trodes_raw_df = trodes_raw_df[trodes_raw_cols].reset_index(drop=True).copy()
    print(trodes_raw_df.head())
    return trodes_raw_df


def make_final_df(trodes_raw_df, trodes_state_df, trodes_video_df):
    """
    Merges the trodes_raw_df and trodes_state_df dataframes and calculates the timestamps for each event.
    Args:
        trodes_raw_df (pandas dataframe): Generated from get_trodes_raw_df.
        trodes_state_df (pandas dataframe): Generated from get_trodes_state_df.
        trodes_video_df (pandas dataframe): Generated from get_trodes_video_df.
    Returns:
        trodes_final_df (pandas dataframe): A dataframe containing the final data for each session.

    """
    trodes_final_df = pd.merge(trodes_raw_df, trodes_state_df, on=["session_dir"], how="inner")
    trodes_final_df = trodes_final_df.rename(columns={"first_item_data": "raw_timestamps"})
    trodes_final_df = trodes_final_df.drop(columns=["metadata_file"], errors="ignore")
    trodes_final_df = trodes_final_df.sort_values(["session_dir", "recording"]).reset_index(drop=True).copy()
    sorted_columns = sorted(trodes_final_df.columns
                            , key=lambda x: x.split("_")[-1])
    trodes_final_df = trodes_final_df[sorted_columns].copy()
    for col in [col for col in trodes_final_df.columns if "timestamps" in col]:
        trodes_final_df[col] = trodes_final_df.apply(lambda x: x[col].astype(np.int32) - np.int32(x["first_timestamp"]),
                                                     axis=1)

    for col in [col for col in trodes_final_df.columns if "frames" in col]:
        trodes_final_df[col] = trodes_final_df[col].apply(lambda x: x.astype(np.int32))

    print("trodes final df")
    print(trodes_final_df.head())
    print(trodes_final_df.columns)
    return trodes_final_df

def merge_state_video_df (trodes_state_df, trodes_video_df):
    """
    Cleans the trodes_state_df and trodes_video_df dataframes and merges them on the session_dir column.
    Args:
        trodes_state_df (pandas dataframe): Generated from get_trodes_state_df.
        trodes_video_df (pandas dataframe): Generated from get_trodes_video_df.
    Returns:
        trodes_state_df (pandas dataframe): A dataframe containing the state data for each session.

    """
    trodes_state_df = pd.merge(trodes_state_df, trodes_video_df, on=["session_dir"], how="inner")
    trodes_state_df["event_frames"] = trodes_state_df.apply(
        lambda x: find_nearest_indices(x["event_timestamps"], x["video_timestamps"]), axis=1)
    print("HERE VIDEO TIME STAMPS")
    print(trodes_state_df["video_timestamps"])
    state_cols_to_keep = ['session_dir', 'metadata_file', 'event_timestamps', 'video_name', 'video_timestamps',
                          'event_frames']
    trodes_state_df = trodes_state_df[state_cols_to_keep].drop_duplicates(
        subset=["session_dir", "metadata_file"]).sort_values(["session_dir", "metadata_file"]).reset_index(
        drop=True).copy()
    same_columns = ['session_dir', 'video_name']
    different_columns = ['metadata_file', 'event_frames', 'event_timestamps']
    trodes_state_df = trodes_state_df.groupby(same_columns).agg(
        {**{col: 'first' for col in trodes_state_df.columns if col not in same_columns + different_columns},
         **{col: lambda x: x.tolist() for col in different_columns}}).reset_index()

    trodes_state_df["tone_timestamps"] = trodes_state_df["event_timestamps"].apply(lambda x: x[0])
    trodes_state_df["port_entry_timestamps"] = trodes_state_df["event_timestamps"].apply(lambda x: x[1])

    trodes_state_df["tone_frames"] = trodes_state_df["event_frames"].apply(lambda x: x[0])
    trodes_state_df["port_entry_frames"] = trodes_state_df["event_frames"].apply(lambda x: x[1])
    trodes_state_df = trodes_state_df.drop(columns=["event_timestamps", "event_frames"], errors="ignore")
    print(trodes_state_df.head())
    return trodes_state_df

def adjust_first_timestamps(trodes_metadata_df, output_dir, experiment_prefix):
    """
    The function will adjust the first timestamps for each session and create a final dataframe containing the
    metadata for each session.
    Args:
        trodes_metadata_df (pandas dataframe): Generated from create_metadata_df.
        output_dir (String): Path to the output directory.
        experiment_prefix (String): Prefix to add to the output files.
    Returns:
        trodes_metadata_df (pandas dataframe): A dataframe containing the metadata for each session.
        trodes_state_df (pandas dataframe): A dataframe containing the state data for each session.
        trodes_video_df (pandas dataframe): A dataframe containing the video data for each session.
        trodes_final_df (pandas dataframe): A dataframe containing the final data for each session.
    """
    trodes_metadata_df = add_subjects_to_metadata(trodes_metadata_df)
    metadata_cols_to_keep = ['raw', 'DIO', 'video_timestamps']
    trodes_metadata_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"].isin(metadata_cols_to_keep)].copy()
    trodes_metadata_df = trodes_metadata_df[~trodes_metadata_df["metadata_file"].str.contains("out")]
    trodes_metadata_df = trodes_metadata_df[~trodes_metadata_df["metadata_file"].str.contains("coordinates")]
    trodes_metadata_df = trodes_metadata_df.reset_index(drop=True)

    trodes_raw_df = trodes_metadata_df[
        (trodes_metadata_df["metadata_dir"] == "raw") & (trodes_metadata_df["metadata_file"] == "timestamps")].copy()

    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(lambda x: x[0])

    recording_to_first_timestamp = trodes_raw_df.set_index('session_dir')['first_timestamp'].to_dict()
    print(recording_to_first_timestamp)
    trodes_metadata_df["first_timestamp"] = trodes_metadata_df["session_dir"].map(recording_to_first_timestamp)
    print(trodes_metadata_df["first_timestamp"])

    trodes_state_df = get_trodes_state_df(trodes_metadata_df)

    trodes_video_df = get_trodes_video_df(trodes_metadata_df)

    trodes_raw_df = get_trodes_raw_df(trodes_metadata_df)

    trodes_state_df = merge_state_video_df(trodes_state_df, trodes_video_df)

    trodes_final_df = make_final_df(trodes_raw_df, trodes_state_df, trodes_video_df)

    # Pickle the final dataframe in the output directory with the experiment prefix.
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # save the final dataframe in experiment path
    pkl_path = os.path.join(output_dir, experiment_prefix + "_final_df.pkl")
    trodes_final_df.to_pickle(pkl_path)
    print("pickle saved in ", os.path.join(pkl_path))

    return trodes_metadata_df, trodes_state_df, trodes_video_df, trodes_final_df, pkl_path


# ADDING TIME Stamps
def load_data(channel_map_path, pickle_path, SUBJECT_COL="Subject"):
    # Load channel mapping
    CHANNEL_MAPPING_DF = pd.read_excel(channel_map_path)
    CHANNEL_MAPPING_DF = CHANNEL_MAPPING_DF.drop(columns=[col for col in CHANNEL_MAPPING_DF.columns if "eib" in col], errors="ignore")
    for col in CHANNEL_MAPPING_DF.columns:
        if "spike_interface" in col:
            CHANNEL_MAPPING_DF[col] = CHANNEL_MAPPING_DF[col].fillna(0)
            CHANNEL_MAPPING_DF[col] = CHANNEL_MAPPING_DF[col].astype(int).astype(str)
    CHANNEL_MAPPING_DF[SUBJECT_COL] = CHANNEL_MAPPING_DF[SUBJECT_COL].astype(str)

    # Load trodes metadata
    SPIKEGADGETS_EXTRACTED_DF = pd.read_pickle(pickle_path)

    return CHANNEL_MAPPING_DF, SPIKEGADGETS_EXTRACTED_DF

def extract_lfp_traces(ALL_SESSION_DIR, ECU_STREAM_ID, TRODES_STREAM_ID, RECORDING_EXTENTION, LFP_FREQ_MIN, LFP_FREQ_MAX, ELECTRIC_NOISE_FREQ, LFP_SAMPLING_RATE, EPHYS_SAMPLING_RATE):
    recording_name_to_all_ch_lfp = {}
    print("ALL SESSION DIR is " + ALL_SESSION_DIR)
    for session_dir in glob.glob(ALL_SESSION_DIR):
        for recording_path in glob.glob(os.path.join(session_dir, RECORDING_EXTENTION)):
            try:
                recording_basename = os.path.splitext(os.path.basename(recording_path))[0]
                current_recording = se.read_spikegadgets(recording_path, stream_id=ECU_STREAM_ID)
                current_recording = se.read_spikegadgets(recording_path, stream_id=TRODES_STREAM_ID)
                print(recording_basename)

                # Preprocessing the LFP
                current_recording = sp.notch_filter(current_recording, freq=ELECTRIC_NOISE_FREQ)
                current_recording = sp.bandpass_filter(current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
                current_recording = sp.resample(current_recording, resample_rate=LFP_SAMPLING_RATE)
                recording_name_to_all_ch_lfp[recording_basename] = current_recording
            except Exception as error:
                print("An exception occurred:", error)
    print("LENGTH OF RECORDING NAME TO ALL CH LFP")
    print(len(recording_name_to_all_ch_lfp))
    return recording_name_to_all_ch_lfp

def combine_lfp_traces_and_metadata(SPIKEGADGETS_EXTRACTED_DF, recording_name_to_all_ch_lfp, CHANNEL_MAPPING_DF, EPHYS_SAMPLING_RATE, LFP_SAMPLING_RATE, LFP_RESAMPLE_RATIO=20, ALL_CH_LFP_COL="all_ch_lfp", SUBJECT_COL="Subject", CURRENT_SUBJECT_COL="current_subject"):

    print("recording name to all channel")
    print(recording_name_to_all_ch_lfp)
    print(SPIKEGADGETS_EXTRACTED_DF.columns)
    lfp_trace_condition = (SPIKEGADGETS_EXTRACTED_DF["recording"].isin(recording_name_to_all_ch_lfp))
    print(lfp_trace_condition)

    SPIKEGADGETS_LFP_DF = SPIKEGADGETS_EXTRACTED_DF[lfp_trace_condition].copy().reset_index(drop=True)
    print("on line 494")
    SPIKEGADGETS_LFP_DF["all_ch_lfp"] = SPIKEGADGETS_LFP_DF["recording"].map(recording_name_to_all_ch_lfp)
    print("on line 496")
    SPIKEGADGETS_LFP_DF["LFP_timestamps"] = SPIKEGADGETS_LFP_DF.apply(
        lambda row: np.arange(0, row["all_ch_lfp"].get_total_samples() * LFP_RESAMPLE_RATIO, LFP_RESAMPLE_RATIO,
                              dtype=int), axis=1)
    print("on line 500")
    SPIKEGADGETS_LFP_DF = pd.merge(SPIKEGADGETS_LFP_DF, CHANNEL_MAPPING_DF, left_on=CURRENT_SUBJECT_COL, right_on=SUBJECT_COL, how="left")
    print("on line 502")
    SPIKEGADGETS_LFP_DF["all_channels"] = SPIKEGADGETS_LFP_DF["all_ch_lfp"].apply(lambda x: x.get_channel_ids())
    SPIKEGADGETS_LFP_DF["region_channels"] = SPIKEGADGETS_LFP_DF[["spike_interface_mPFC", "spike_interface_vHPC", "spike_interface_BLA", "spike_interface_LH", "spike_interface_MD"]].to_dict('records')
    SPIKEGADGETS_LFP_DF["region_channels"] = SPIKEGADGETS_LFP_DF["region_channels"].apply(lambda x: sorted(x.items(), key=lambda item: int(item[1])))
    print(SPIKEGADGETS_LFP_DF["region_channels"].iloc[0])
    print("on line 506")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    print(SPIKEGADGETS_LFP_DF.head())
    print(SPIKEGADGETS_LFP_DF.columns)
    def get_traces_with_progress(row):
        channel_ids = [t[1] for t in row["region_channels"]]
        total_channels = len(channel_ids)
        logging.info(f"Processing {total_channels} channels for row {row.name}")

        traces = row[ALL_CH_LFP_COL].get_traces(channel_ids=channel_ids)
        logging.info(f"Completed processing channels for row {row.name}")

        return traces.T
    #print number of rows in SPIKEGADGETS_LFP_DF
    print("num rows in spike df")
    print(len(SPIKEGADGETS_LFP_DF))
    # Apply the modified function
    SPIKEGADGETS_LFP_DF["all_region_lfp_trace"] = SPIKEGADGETS_LFP_DF.apply(get_traces_with_progress, axis=1)
    print("on line 508")
    SPIKEGADGETS_LFP_DF["per_region_lfp_trace"] = SPIKEGADGETS_LFP_DF.apply(lambda row: dict(zip(["{}_lfp_trace".format(t[0].strip("spike_interface_")) for t in row["region_channels"]], row["all_region_lfp_trace"])), axis=1)
    SPIKEGADGETS_FINAL_DF = pd.concat([SPIKEGADGETS_LFP_DF.copy(), SPIKEGADGETS_LFP_DF['per_region_lfp_trace'].apply(pd.Series).copy()], axis=1)
    print("on line 510")
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF.drop(columns=["all_channels", "all_region_lfp_trace", "per_region_lfp_trace", "region_channels", "all_ch_lfp"], errors="ignore")
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF.drop(columns=[col for col in SPIKEGADGETS_FINAL_DF.columns if "spike_interface" in col], errors="ignore")
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF.rename(columns={col: col.lower() for col in SPIKEGADGETS_LFP_DF.columns})
    sorted_columns = sorted(SPIKEGADGETS_FINAL_DF.columns, key=lambda x: x.split("_")[-1])
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF[sorted_columns].copy()

    print("done combining lfp traces and metadata")

    return SPIKEGADGETS_FINAL_DF

### START OF NOTEBOOK 2 ###

def generate_pairs(lst):
    """
    Generates all unique pairs from a list.

    Parameters:
    - lst (list): The list to generate pairs from.

    Returns:
    - list: A list of tuples, each containing a unique pair from the input list.
    """
    n = len(lst)
    return [(lst[i], lst[j]) for i in range(n) for j in range(i+1, n)]

def update_array_by_mask(array, mask, value=np.nan):
    """
    Update elements of an array based on a mask and replace them with a specified value.

    Parameters:
    - array (np.array): The input numpy array whose values are to be updated.
    - mask (np.array): A boolean array with the same shape as `array`. Elements of `array` corresponding to True in the mask are replaced.
    - value (scalar, optional): The value to assign to elements of `array` where `mask` is True. Defaults to np.nan.

    Returns:
    - np.array: A copy of the input array with updated values where the mask is True.

    Example:
    array = np.array([1, 2, 3, 4])
    mask = np.array([False, True, False, True])
    update_array_by_mask(array, mask, value=0)
    array([1, 0, 3, 0])
    """
    result = array.copy()
    result[mask] = value
    return result

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        # linear interpolation of NaNs
        nans, x= nan_helper(y)
        y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def preprocess_lfp_data(lfp_traces_df, voltage_scaling_value, zscore_threshold, resample_rate):
    print("beginning preprocessing")
    original_trace_columns = [col for col in lfp_traces_df.columns if "trace" in col]

    for col in original_trace_columns:
        lfp_traces_df[col] = lfp_traces_df[col].apply(lambda x: x.astype(np.float32) * voltage_scaling_value)

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_MAD".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: stats.median_abs_deviation(x))

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_modified_zscore".format(brain_region)
        MAD_column = "{}_lfp_MAD".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df.apply(lambda x: 0.6745 * (x[col] - np.median(x[col])) / x[MAD_column], axis=1)

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_RMS".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: (x / np.sqrt(np.mean(x**2))).astype(np.float32))

    zscore_columns = [col for col in lfp_traces_df.columns if "zscore" in col]
    for col in zscore_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_mask".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: np.abs(x) >= zscore_threshold)

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_trace_filtered".format(brain_region)
        mask_column = "{}_lfp_mask".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df.apply(lambda x: update_array_by_mask(x[col], x[mask_column]), axis=1)

    filtered_trace_column = [col for col in lfp_traces_df if "lfp_trace_filtered" in col]
    for col in filtered_trace_column:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_RMS_filtered".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: (x / np.sqrt(np.nanmean(x**2))).astype(np.float32))
    print("done preprocessing")
    return lfp_traces_df

def modified_z_score(original_trace_columns, LFP_TRACES_DF, zscore_threshold=4):
    print("Calculating modified z-score")
    for col in original_trace_columns:
        print(col)
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_MAD".format(brain_region)
        LFP_TRACES_DF[updated_column] = LFP_TRACES_DF[col].apply(lambda x: stats.median_abs_deviation(x))

    for col in original_trace_columns:
        print(col)
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_modified_zscore".format(brain_region)
        MAD_column = "{}_lfp_MAD".format(brain_region)

        LFP_TRACES_DF[updated_column] = LFP_TRACES_DF.apply(
            lambda x: 0.6745 * (x[col] - np.median(x[col])) / x[MAD_column], axis=1)

    # root-mean-square
    for col in original_trace_columns:
        print(col)
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_RMS".format(brain_region)
        LFP_TRACES_DF[updated_column] = LFP_TRACES_DF[col].apply(
            lambda x: (x / np.sqrt(np.mean(x ** 2))).astype(np.float32))

    zscore_columns = [col for col in LFP_TRACES_DF.columns if "zscore" in col]

    for col in zscore_columns:
        print(col)
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_mask".format(brain_region)
        LFP_TRACES_DF[updated_column] = LFP_TRACES_DF[col].apply(lambda x: np.abs(x) >= zscore_threshold)

    for col in original_trace_columns:
        print(col)
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_trace_filtered".format(brain_region)
        mask_column = "{}_lfp_mask".format(brain_region)
        LFP_TRACES_DF[updated_column] = LFP_TRACES_DF.apply(lambda x: update_array_by_mask(x[col], x[mask_column]),
                                                            axis=1)

    filtered_trace_column = [col for col in LFP_TRACES_DF if "lfp_trace_filtered" in col]
    for col in filtered_trace_column:
        print(col)
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_RMS_filtered".format(brain_region)
        LFP_TRACES_DF[updated_column] = LFP_TRACES_DF[col].apply(
            lambda x: (x / np.sqrt(np.nanmean(x ** 2))).astype(np.float32))

    print(LFP_TRACES_DF.head())
    print(original_trace_columns.head())

    return LFP_TRACES_DF, original_trace_columns

def calculate_power(lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step):
    print("calculating power")
    input_columns = [col for col in lfp_traces_df.columns if "trace" in col or "RMS" in col]

    for col in input_columns:
        brain_region = col.replace("_lfp", "")

        multitaper_col = f"{brain_region}_power_multitaper"
        connectivity_col = f"{brain_region}_power_connectivity"
        frequencies_col = f"{brain_region}_power_frequencies"
        power_col = f"{brain_region}_power_all_frequencies_all_windows"

        try:
            lfp_traces_df[multitaper_col] = lfp_traces_df[col].apply(
                lambda x: Multitaper(
                    time_series=x,
                    sampling_frequency=resample_rate,
                    time_halfbandwidth_product=time_halfbandwidth_product,
                    time_window_duration=time_window_duration,
                    time_window_step=time_window_step
                )
            )

            lfp_traces_df[connectivity_col] = lfp_traces_df[multitaper_col].apply(
                lambda x: Connectivity.from_multitaper(x)
            )

            lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                lambda x: x.frequencies
            )
            lfp_traces_df[power_col] = lfp_traces_df[connectivity_col].apply(
                lambda x: x.power().squeeze()
            )

            lfp_traces_df[power_col] = lfp_traces_df[power_col].apply(lambda x: x.astype(np.float16))

            lfp_traces_df = lfp_traces_df.drop(columns=[multitaper_col, connectivity_col], errors="ignore")

        except Exception as e:
            print(e)

    lfp_traces_df["power_timestamps"] = lfp_traces_df["lfp_timestamps"].apply(lambda x: x[(resample_rate//2):(-resample_rate//2):(resample_rate//2)])
    lfp_traces_df["power_calculation_frequencies"] = lfp_traces_df[[col for col in lfp_traces_df.columns if "power_frequencies" in col][0]].copy()
    lfp_traces_df = lfp_traces_df.drop(columns=[col for col in lfp_traces_df.columns if "power_frequencies" in col], errors="ignore")

    return lfp_traces_df

def calculate_phase(lfp_traces_df, fs):
    print("calculating phase")
    from scipy.signal import butter, filtfilt, hilbert

    order = 4
    RMS_columns = [col for col in lfp_traces_df if "RMS" in col and "filtered" in col and "all" not in col]

    # Filter for theta band
    freq_band = [4, 12]
    b, a = butter(order, freq_band, fs=fs, btype='band')
    for col in RMS_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_theta_band".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: filtfilt(b, a, x, padtype=None).astype(np.float32))

    # Filter for gamma band
    freq_band = [30, 50]
    b, a = butter(order, freq_band, fs=fs, btype='band')
    for col in RMS_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_gamma_band".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: filtfilt(b, a, x, padtype=None).astype(np.float32))

    # Calculate phase
    band_columns = [col for col in lfp_traces_df if "band" in col]
    for col in band_columns:
        brain_region = col.replace("_band", "")
        updated_column = "{}_phase".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(lambda x: np.angle(hilbert(x), deg=False))

    return lfp_traces_df

def calculate_coherence(lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step):
    print("calculating coherence")
    input_columns = [col for col in lfp_traces_df.columns if "trace" in col or "RMS" in col]
    all_suffixes = set(["_".join(col.split("_")[1:]) for col in input_columns])
    brain_region_pairs = generate_pairs(list(set([col.split("lfp")[0].strip("_") for col in input_columns])))

    for first_region, second_region in brain_region_pairs:
        for suffix in all_suffixes:
            suffix_for_name = suffix.replace("lfp", "").strip("_")
            region_1 = "_".join([first_region, suffix])
            region_2 = "_".join([second_region, suffix])
            pair_base_name = f"{region_1.split('_')[0]}_{region_2.split('_')[0]}_{suffix_for_name}"

            try:
                multitaper_col = f"{pair_base_name}_coherence_multitaper"
                connectivity_col = f"{pair_base_name}_coherence_connectivity"
                frequencies_col = f"{pair_base_name}_coherence_frequencies"
                coherence_col = f"{pair_base_name}_coherence_all_frequencies_all_windows"

                lfp_traces_df[multitaper_col] = lfp_traces_df.apply(
                    lambda x: Multitaper(
                        time_series=np.array([x[region_1], x[region_2]]).T,
                        sampling_frequency=resample_rate,
                        time_halfbandwidth_product=time_halfbandwidth_product,
                        time_window_step=time_window_step,
                        time_window_duration=time_window_duration
                    ),
                    axis=1
                )

                lfp_traces_df[connectivity_col] = lfp_traces_df[multitaper_col].apply(
                    lambda x: Connectivity.from_multitaper(x)
                )

                lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.frequencies
                )
                lfp_traces_df[coherence_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.coherence_magnitude()[:,:,0,1]
                )

                lfp_traces_df[coherence_col] = lfp_traces_df[coherence_col].apply(lambda x: x.astype(np.float16))

            except Exception as e:
                print(e)

            lfp_traces_df = lfp_traces_df.drop(columns=[multitaper_col, connectivity_col], errors="ignore")

    lfp_traces_df["coherence_timestamps"] = lfp_traces_df["lfp_timestamps"].apply(lambda x: x[(resample_rate//2):(-resample_rate//2):(resample_rate//2)])
    lfp_traces_df["coherence_calculation_frequencies"] = lfp_traces_df[[col for col in lfp_traces_df.columns if "coherence_frequencies" in col][0]].copy()
    lfp_traces_df = lfp_traces_df.drop(columns=[col for col in lfp_traces_df.columns if "coherence_frequencies" in col], errors="ignore")

    return lfp_traces_df

def calculate_granger_causality(lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step):
    print("calculating granger causality")
    input_columns = [col for col in lfp_traces_df.columns if "trace" in col or "RMS" in col]
    all_suffixes = set(["_".join(col.split("_")[1:]) for col in input_columns])
    brain_region_pairs = generate_pairs(list(set([col.split("lfp")[0].strip("_") for col in input_columns])))

    for first_region, second_region in brain_region_pairs:
        for suffix in all_suffixes:
            region_1 = "_".join([first_region, suffix])
            region_2 = "_".join([second_region, suffix])
            region_1_base_name = region_1.split('_')[0]
            region_2_base_name = region_2.split('_')[0]
            pair_base_name = f"{region_1_base_name}_{region_2_base_name}"

            try:
                multitaper_col = f"{pair_base_name}_granger_multitaper"
                connectivity_col = f"{pair_base_name}_granger_connectivity"
                frequencies_col = f"{pair_base_name}_granger_frequencies"
                granger_1_2_col = f"{region_1_base_name}_{region_2_base_name}_granger_all_frequencies_all_windows"
                granger_2_1_col = f"{region_2_base_name}_{region_1_base_name}_granger_all_frequencies_all_windows"

                lfp_traces_df[multitaper_col] = lfp_traces_df.apply(
                    lambda x: Multitaper(
                        time_series=np.array([x[region_1], x[region_2]]).T,
                        sampling_frequency=resample_rate,
                        time_halfbandwidth_product=time_halfbandwidth_product,
                        time_window_step=time_window_step,
                        time_window_duration=time_window_duration
                    ),
                    axis=1
                )

                lfp_traces_df[connectivity_col] = lfp_traces_df[multitaper_col].apply(
                    lambda x: Connectivity.from_multitaper(x)
                )

                lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.frequencies
                )

                lfp_traces_df[granger_1_2_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.pairwise_spectral_granger_prediction()[:,:,0,1]
                )

                lfp_traces_df[granger_2_1_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.pairwise_spectral_granger_prediction()[:,:,1,0]
                )

                lfp_traces_df[granger_1_2_col] = lfp_traces_df[granger_1_2_col].apply(lambda x: x.astype(np.float16))
                lfp_traces_df[granger_2_1_col] = lfp_traces_df[granger_2_1_col].apply(lambda x: x.astype(np.float16))

            except Exception as e:
                print(e)

            lfp_traces_df = lfp_traces_df.drop(columns=[multitaper_col, connectivity_col], errors="ignore")

    lfp_traces_df["granger_timestamps"] = lfp_traces_df["lfp_timestamps"].apply(lambda x: x[(resample_rate//2):(-resample_rate//2):(resample_rate//2)])
    lfp_traces_df["granger_calculation_frequencies"] = lfp_traces_df[[col for col in lfp_traces_df.columns if "granger_frequencies" in col][0]].copy()
    lfp_traces_df = lfp_traces_df.drop(columns=[col for col in lfp_traces_df.columns if "granger_frequencies" in col], errors="ignore")

    return lfp_traces_df

def main_test_only():
    input_dir = "/Volumes/chaitra/reward_competition_extension/data/standard/2023_06_*/*.rec"
    output_dir = "/Volumes/chaitra/reward_competition_extension/data/proc/"
    channel_map_path = "channel_mapping.xlsx"
    TONE_DIN = "dio_ECU_Din1"
    TONE_STATE = 1
    experiment_dir = "/Volumes/chaitra/reward_competition_extension/data"
    experiment_prefix = "rce_test"
    #convert_to_mp4(experiment_dir)
    paths = {}
    #session_to_trodes_temp, paths= extract_all_trodes(input_dir)
    #session_to_trodes_temp = add_video_timestamps(session_to_trodes_temp, input_dir)
    #metadata = create_metadata_df(session_to_trodes_temp, paths)
    #metadata, state_df, video_df, final_df, pkl_path = adjust_first_timestamps(metadata, output_dir, experiment_prefix)

    print("output from obj creation")


    # try to create LFPObject
    lfp = LFPObject(path=input_dir, channel_map_path=channel_map_path, events_path="test.xlsx", subject="1.4")

    #write out all the dataframes to a text file
    lfp.metadata.to_csv("test_outputs/metadata.txt", sep="\t")
    lfp.state_df.to_csv("test_outputs/state_df.txt", sep="\t")
    lfp.video_df.to_csv("test_outputs/video_df.txt", sep="\t")
    lfp.final_df.to_csv("test_outputs/final_df.txt", sep="\t")
    lfp.power_df.to_csv("test_outputs/power_df.txt", sep="\t")
    lfp.phase_df.to_csv("test_outputs/phase_df.txt", sep="\t")
    lfp.coherence_df.to_csv("test_outputs/coherence_df.txt", sep="\t")
    lfp.granger_df.to_csv("test_outputs/granger_df.txt", sep="\t")


main_test_only()