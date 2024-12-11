# %%
import h5py
import pandas
import numpy
import os
import glob
from collections import defaultdict
import trodes.read_exported
import pandas as pd
import numpy as np
from scipy import stats
from spectral_connectivity import Multitaper, Connectivity
import logging
import h5py
import pickle
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
from multiprocessing import Process

import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp


def pickle_this(thing_to_pickle, file_name):
    """
    Pickles things
    Args (2):   
        thing_to_pickle: anything you want to pickle
        file_name: str, filename that ends with .pkl 
    Returns:
        none
    """
    with open(file_name,'wb') as file:
        pickle.dump(thing_to_pickle, file)

def unpickle_this(pickle_file):
    """
    Unpickles things
    Args (1):   
        file_name: str, pickle filename that already exists and ends with .pkl
    Returns:
        pickled item
    """
    with open(pickle_file, 'rb') as file:
        return(pickle.load(file))

# %% [markdown]
# Questions: how are the sleap files and the lfp files matched up? do they need the same names? yes but the h5 files are not used? 
# 
      

# %%
def helper_generate_pairs(lst):
    """
    Generates all unique pairs from a list.

    Parameters:
    - lst (list): The list to generate pairs from.

    Returns:
    - list: A list of tuples, each containing a unique pair from the input list.
    """
    n = len(lst)
    return [(lst[i], lst[j]) for i in range(n) for j in range(i + 1, n)]

def helper_update_array_by_mask(array, mask, value=np.nan):
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

def helper_find_nearest_indices(array1, array2):
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

def create_metadata_df(session_to_trodes, session_to_path):
    """

    Args:
        session_to_trodes (nested dictionary): Generated from extract_all_trodes.
        session_to_path (empty dictionary): {}
        columns_to_keep (dictionary): Provide a dictionary of the columns to keep in the metadata dataframe.

    Returns:
        trodes_metadata_df (pandas dataframe): A dataframe containing the metadata for each session.
    """

    trodes_metadata_df = pd.DataFrame.from_dict({(i, j, k, l): session_to_trodes[i][j][k][l]
                                                 for i in session_to_trodes.keys()
                                                 for j in session_to_trodes[i].keys()
                                                 for k in session_to_trodes[i][j].keys()
                                                 for l in session_to_trodes[i][j][k].keys()},
                                                orient='index')

    trodes_metadata_df = trodes_metadata_df.reset_index()
    trodes_metadata_df = trodes_metadata_df.rename(
        columns={
            'level_0': 'session_dir',
            'level_1': 'recording',
            'level_2': 'metadata_dir',
            'level_3': 'metadata_file'},
        errors="ignore")
    trodes_metadata_df["session_path"] = trodes_metadata_df["session_dir"].map(
        session_to_path)

    # Adjust data types
    trodes_metadata_df["first_dtype_name"] = trodes_metadata_df["data"].apply(
        lambda x: x.dtype.names[0])
    trodes_metadata_df["first_item_data"] = trodes_metadata_df["data"].apply(
        lambda x: x[x.dtype.names[0]])
    trodes_metadata_df["last_dtype_name"] = trodes_metadata_df["data"].apply(
        lambda x: x.dtype.names[-1])
    trodes_metadata_df["last_item_data"] = trodes_metadata_df["data"].apply(
        lambda x: x[x.dtype.names[-1]])

    ##print("unique recordings ")
    #print(trodes_metadata_df["recording"].unique())
    return trodes_metadata_df



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

    for session in glob.glob(input_dir):
        try:
            session_basename = os.path.splitext(os.path.basename(session))[0]
            #print("Processing session: ", session_basename)
            session_to_trodes_data[session_basename] = trodes.read_exported.organize_all_trodes_export(
                session)
            session_to_path[session_basename] = session
        except Exception as e:
            print("Error processing session: ", session_basename)
            print(e)

    # #print(session_to_trodes_data)
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
            #print("Current Session: {}".format(session_basename))

            for video_timestamps in glob.glob(
                    os.path.join(session, "*cameraHWSync")):
                video_basename = os.path.basename(video_timestamps)
                #print("Current Video Name: {}".format(video_basename))
                timestamp_array = trodes.read_exported.read_trodes_extracted_data_file(
                    video_timestamps)

                if "video_timestamps" not in session_to_trodes_data[session_basename][session_basename]:
                    session_to_trodes_data[session_basename][session_basename]["video_timestamps"] = defaultdict(
                        dict)

                session_to_trodes_data[session_basename][session_basename]["video_timestamps"][video_basename.split(
                    ".")[-3]] = timestamp_array
                #print("Timestamp Array for {}: ".format(video_basename))
                #print(session_to_trodes_data[session_basename][session_basename]["video_timestamps"][video_basename.split(".")[-3]])

        except Exception as e:
            print("Error processing session: ", session_basename)
            print(e)
           
        return session_to_trodes_data
    
def adjust_first_timestamps(trodes_metadata_df):
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
    trodes_metadata_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"].isin(
        metadata_cols_to_keep)].copy()
    trodes_metadata_df = trodes_metadata_df[~trodes_metadata_df["metadata_file"].str.contains(
        "out")]
    trodes_metadata_df = trodes_metadata_df[~trodes_metadata_df["metadata_file"].str.contains(
        "coordinates")]
    trodes_metadata_df = trodes_metadata_df.reset_index(drop=True)

    trodes_raw_df = trodes_metadata_df[(trodes_metadata_df["metadata_dir"] == "raw") & (
        trodes_metadata_df["metadata_file"] == "timestamps")].copy()

    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(
        lambda x: x[0])

    recording_to_first_timestamp = trodes_raw_df.set_index(
        'session_dir')['first_timestamp'].to_dict()
    #print(recording_to_first_timestamp)
    trodes_metadata_df["first_timestamp"] = trodes_metadata_df["session_dir"].map(
        recording_to_first_timestamp)
    #print(trodes_metadata_df["first_timestamp"])

    #trodes_state_df = get_trodes_state_df(trodes_metadata_df)

    trodes_video_df = get_trodes_video_df(trodes_metadata_df)

    #trodes_state_df = merge_state_video_df(trodes_state_df, trodes_video_df)

    return trodes_metadata_df, trodes_video_df

def get_trodes_video_df(trodes_metadata_df):
    """
    Extracts the video data from the trodes_metadata_df and calculates the first timestamp for each session.
    Args:
        trodes_metadata_df (pandas dataframe): Generated from create_metadata_df.
    Returns:
        trodes_video_df (pandas dataframe): A dataframe containing the video data for each session.
    """
    trodes_video_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"]
                                         == "video_timestamps"].copy().reset_index(drop=True)
    trodes_video_df = trodes_video_df[trodes_video_df["metadata_file"] == "1"].copy(
    )
    trodes_video_df["video_timestamps"] = trodes_video_df["first_item_data"]
    trodes_video_df = trodes_video_df[[
        "filename", "video_timestamps", "session_dir"]].copy()
    trodes_video_df = trodes_video_df.rename(
        columns={"filename": "video_name"})
    #print(trodes_video_df.head())
    return trodes_video_df


def get_trodes_raw_df(trodes_metadata_df):
    """
    Extracts the raw data from the trodes_metadata_df and calculates the first timestamp for each session.
    Args:
        trodes_metadata_df (pandas dataframe): Generated from create_metadata_df.
    Returns:
        trodes_raw_df (pandas dataframe): A dataframe containing the raw data for each session.
    """
    trodes_raw_df = trodes_metadata_df[(trodes_metadata_df["metadata_dir"] == "raw") & (
        trodes_metadata_df["metadata_file"] == "timestamps")].copy()
    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(
        lambda x: x[0])
    trodes_raw_cols = [
        'session_dir',
        'recording',
        'original_file',
        'session_path',
        'current_subject',
        'first_item_data',
        'first_timestamp',
        'all_subjects']
    trodes_raw_df = trodes_raw_df[trodes_raw_cols].reset_index(
        drop=True).copy()
    #print(trodes_raw_df.head())
    return trodes_raw_df


def add_subjects_to_metadata(metadata):
    """
    Adds the subjects to the metadata dataframe.
    Args:
        metadata (pandas dataframe): Generated from create_metadata_df.
    Returns:
        metadata (pandas dataframe): A dataframe containing the metadata for each session with the subjects added.
    """
    # TODO: find a better way to do this without regex on the session_dir
    metadata["all_subjects"] = metadata["session_dir"].apply(lambda x: x.replace(
        "-", "_").split("subj")[-1].split("t")[0].strip("_").replace("_", ".").split(".and."))
    metadata["all_subjects"] = metadata["all_subjects"].apply(
        lambda x: sorted([i.strip().strip(".") for i in x]))
    metadata["current_subject"] = metadata["session_dir"].apply(lambda x: x.split('_')[2])
    #print(metadata["all_subjects"])
    #print(metadata["current_subject"])
    return metadata


class LfpRecordingObject:
    def __init__(self, path, output_dir):
        # call extract_all_trodes
        session_to_trodes_temp, paths = extract_all_trodes(input_dir=path)
        # call add_video_timestamps
        session_to_trodes_temp = add_video_timestamps(
            session_to_trodes_data=session_to_trodes_temp,
            directory_path=path)
        # call create_metadata_df
        metadata = create_metadata_df(
            session_to_trodes=session_to_trodes_temp, session_to_path=paths)
        # call adjust_first_timestamps
        self.metadata, self.video_df= adjust_first_timestamps(
            trodes_metadata_df=metadata)

        #print(output_dir)

        # Pickle all
        self.metadata.to_pickle(output_dir + "/metadata.pkl")
        self.video_df.to_pickle(output_dir + "/video_df.pkl")

        #print("LFP Object has been created for" + path)


def load_channel_map(channel_map_path, SUBJECT_COL="Subject"):
    """
    Loads the channel mapping and trodes metadata dataframe.
    Args:
        channel_map_path (String): Path to the channel mapping excel file.
        SUBJECT_COL (String): Column name for the subject in the channel mapping dataframe.
    Returns:
        CHANNEL_MAPPING_DF (pandas dataframe): A dataframe containing the channel mapping data.
    """
    # Load channel mapping
    CHANNEL_MAPPING_DF = pd.read_excel(channel_map_path)
    CHANNEL_MAPPING_DF = CHANNEL_MAPPING_DF.drop(
        columns=[
            col for col in CHANNEL_MAPPING_DF.columns if "eib" in col],
        errors="ignore")
    for col in CHANNEL_MAPPING_DF.columns:
        if "spike_interface" in col:
            CHANNEL_MAPPING_DF[col] = CHANNEL_MAPPING_DF[col].fillna(0)
            CHANNEL_MAPPING_DF[col] = CHANNEL_MAPPING_DF[col].astype(
                int).astype(str)
    CHANNEL_MAPPING_DF[SUBJECT_COL] = CHANNEL_MAPPING_DF[SUBJECT_COL].astype(
        str)

    return CHANNEL_MAPPING_DF

def extract_lfp_traces(
        ALL_SESSION_DIR,
        ECU_STREAM_ID,
        TRODES_STREAM_ID,
        RECORDING_EXTENTION,
        LFP_FREQ_MIN,
        LFP_FREQ_MAX,
        ELECTRIC_NOISE_FREQ = 60,
        LFP_SAMPLING_RATE=1000,
        EPHYS_SAMPLING_RATE = 20000):
    """
    Extracts the LFP traces from the SpikeGadgets recordings using the spikeextractors module.
    Args:
        ALL_SESSION_DIR (String): Path to the directory containing the session directories.
        ECU_STREAM_ID (String): The stream ID for the ECU data.
        TRODES_STREAM_ID (String): The stream ID for the trodes data.
        RECORDING_EXTENTION (String): The file extension for the recordings.
        LFP_FREQ_MIN (float): The minimum frequency for the LFP bandpass filter.
        LFP_FREQ_MAX (float): The maximum frequency for the LFP bandpass filter.
        ELECTRIC_NOISE_FREQ (float): The frequency of the electric noise.
        LFP_SAMPLING_RATE (int): The sampling rate for the LFP traces.
        EPHYS_SAMPLING_RATE (int): The sampling rate for the ephys traces.
    Returns:
        recording_name_to_all_ch_lfp (dictionary): A dictionary containing the LFP traces for each recording.
    """
    recording_name_to_all_ch_lfp = {}
    #print("ALL SESSION DIR is " + ALL_SESSION_DIR)
    for session_dir in glob.glob(ALL_SESSION_DIR):
        for recording_path in glob.glob(
                os.path.join(session_dir, RECORDING_EXTENTION)):
            try:
                recording_basename = os.path.splitext(
                    os.path.basename(recording_path))[0]
                current_recording = se.read_spikegadgets(
                    recording_path, stream_id=TRODES_STREAM_ID)
                #print(recording_basename)

                # Preprocessing the LFP
                current_recording = sp.notch_filter(
                    current_recording, freq=ELECTRIC_NOISE_FREQ)
                current_recording = sp.bandpass_filter(
                    current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
                current_recording = sp.resample(
                    current_recording, resample_rate=LFP_SAMPLING_RATE)
                recording_name_to_all_ch_lfp[recording_basename] = current_recording
            except Exception as error:
                print("An exception occurred:", error)
              
    #print("LENGTH OF RECORDING NAME TO ALL CH LFP")
    #print(len(recording_name_to_all_ch_lfp))
    return recording_name_to_all_ch_lfp



def combine_lfp_traces_and_metadata(
        SPIKEGADGETS_EXTRACTED_DF,
        recording_name_to_all_ch_lfp,
        CHANNEL_MAPPING_DF,
        EPHYS_SAMPLING_RATE = 20000,
        LFP_SAMPLING_RATE = 1000,
        LFP_RESAMPLE_RATIO=20,
        ALL_CH_LFP_COL="all_ch_lfp",
        SUBJECT_COL="Subject",
        CURRENT_SUBJECT_COL="current_subject"):
    """
    Combines the LFP traces with the metadata in the SpikeGadgets dataframe.
    Args:
        SPIKEGADGETS_EXTRACTED_DF (pandas dataframe): A dataframe containing the trodes metadata.
        recording_name_to_all_ch_lfp (dictionary): A dictionary containing the LFP traces for each recording.
        CHANNEL_MAPPING_DF (pandas dataframe): A dataframe containing the channel mapping data.
        EPHYS_SAMPLING_RATE (int): The sampling rate for the ephys traces.
        LFP_SAMPLING_RATE (int): The sampling rate for the LFP traces.
        LFP_RESAMPLE_RATIO (int): The ratio to resample the LFP traces.
        ALL_CH_LFP_COL (String): The column name for the LFP traces in the SpikeGadgets dataframe.
        SUBJECT_COL (String): Column name for the subject in the channel mapping dataframe.
        CURRENT_SUBJECT_COL (String): Column name for the current subject in the SpikeGadgets dataframe.
    Returns:
        SPIKEGADGETS_FINAL_DF (pandas dataframe): A dataframe containing the final data for each session.
    """
    #print("recording name to all channel")
    #print(recording_name_to_all_ch_lfp)
    #print(SPIKEGADGETS_EXTRACTED_DF.columns)
    lfp_trace_condition = (
        SPIKEGADGETS_EXTRACTED_DF["recording"].isin(recording_name_to_all_ch_lfp))
    #print(lfp_trace_condition)

    SPIKEGADGETS_LFP_DF = SPIKEGADGETS_EXTRACTED_DF[lfp_trace_condition].copy(
    ).reset_index(drop=True)
    #print("on line 494")
    SPIKEGADGETS_LFP_DF["all_ch_lfp"] = SPIKEGADGETS_LFP_DF["recording"].map(
        recording_name_to_all_ch_lfp)
    #print("on line 496")
    SPIKEGADGETS_LFP_DF["LFP_timestamps"] = SPIKEGADGETS_LFP_DF.apply(
        lambda row: np.arange(
            0,
            row["all_ch_lfp"].get_total_samples() *
            LFP_RESAMPLE_RATIO,
            LFP_RESAMPLE_RATIO,
            dtype=int),
        axis=1)
    #print("on line 500")
    SPIKEGADGETS_LFP_DF = pd.merge(
        SPIKEGADGETS_LFP_DF,
        CHANNEL_MAPPING_DF,
        left_on=CURRENT_SUBJECT_COL,
        right_on=SUBJECT_COL,
        how="left")
    #print("on line 502")
    SPIKEGADGETS_LFP_DF["all_channels"] = SPIKEGADGETS_LFP_DF["all_ch_lfp"].apply(
        lambda x: x.get_channel_ids())
    SPIKEGADGETS_LFP_DF["region_channels"] = SPIKEGADGETS_LFP_DF[["spike_interface_mPFC",
                                                                  "spike_interface_vHPC",
                                                                  "spike_interface_BLA",
                                                                  "spike_interface_LH",
                                                                  "spike_interface_MD"]].to_dict('records')
    SPIKEGADGETS_LFP_DF["region_channels"] = SPIKEGADGETS_LFP_DF["region_channels"].apply(
        lambda x: sorted(x.items(), key=lambda item: int(item[1])))
    #print(SPIKEGADGETS_LFP_DF["region_channels"].iloc[0])
    #print("on line 506")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    #print(SPIKEGADGETS_LFP_DF.head())
    #print(SPIKEGADGETS_LFP_DF.columns)

    def get_traces_with_progress(row):
        """
        Extracts the LFP traces for each region in the input row.
        Args:
            row (pandas series): A row in the SpikeGadgets dataframe.
        Returns:
            traces (numpy array): A numpy array containing the LFP traces for each region.
        """
        channel_ids = [t[1] for t in row["region_channels"]]
        total_channels = len(channel_ids)
        logging.info(
            f"Processing {total_channels} channels for row {row.name}")

        traces = row[ALL_CH_LFP_COL].get_traces(channel_ids=channel_ids)
        logging.info(f"Completed processing channels for row {row.name}")

        return traces.T

    # #print number of rows in SPIKEGADGETS_LFP_DF
    #print("num rows in spike df")
    #print(len(SPIKEGADGETS_LFP_DF))
    # Apply the modified function
    SPIKEGADGETS_LFP_DF["all_region_lfp_trace"] = SPIKEGADGETS_LFP_DF.apply(
        get_traces_with_progress, axis=1)
    #print("on line 508")
    SPIKEGADGETS_LFP_DF["per_region_lfp_trace"] = SPIKEGADGETS_LFP_DF.apply(lambda row: dict(zip(["{}_lfp_trace".format(
        t[0].strip("spike_interface_")) for t in row["region_channels"]], row["all_region_lfp_trace"])), axis=1)
    SPIKEGADGETS_FINAL_DF = pd.concat([SPIKEGADGETS_LFP_DF.copy(
    ), SPIKEGADGETS_LFP_DF['per_region_lfp_trace'].apply(pd.Series).copy()], axis=1)
    #print("on line 510")
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF.drop(
        columns=[
            "all_channels",
            "all_region_lfp_trace",
            "per_region_lfp_trace",
            "region_channels",
            "all_ch_lfp"],
        errors="ignore")
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF.drop(
        columns=[
            col for col in SPIKEGADGETS_FINAL_DF.columns if "spike_interface" in col],
        errors="ignore")
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF.rename(
        columns={col: col.lower() for col in SPIKEGADGETS_LFP_DF.columns})
    sorted_columns = sorted(SPIKEGADGETS_FINAL_DF.columns,
                            key=lambda x: x.split("_")[-1])
    SPIKEGADGETS_FINAL_DF = SPIKEGADGETS_FINAL_DF[sorted_columns].copy()

    #print("done combining lfp traces and metadata")

    return SPIKEGADGETS_FINAL_DF

def preprocess_lfp_data(lfp_traces_df, voltage_scaling_value, zscore_threshold, resample_rate):
    """
    Preprocesses the LFP traces in the input dataframe by calculating the modified z-score, root-mean-square, and filtering out outliers.
    Args:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces.
        voltage_scaling_value (float): The scaling value for the LFP traces.
        zscore_threshold (float): The threshold for the modified z-score.
        resample_rate (int): The resample rate for the LFP traces.
    Returns:
        lfp_traces_df (pandas dataframe): A dataframe containing the preprocessed LFP traces.
    """
    #print("beginning preprocessing")
    original_trace_columns = [
        col for col in lfp_traces_df.columns if "trace" in col]

    for col in original_trace_columns:
        lfp_traces_df[col] = lfp_traces_df[col].apply(
            lambda x: x.astype(np.float32) * voltage_scaling_value)

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_MAD".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: stats.median_abs_deviation(x))

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_modified_zscore".format(brain_region)
        MAD_column = "{}_lfp_MAD".format(brain_region)
        SPIKE_GADGETS_MULTIPLIER = 0.6745
        lfp_traces_df[updated_column] = lfp_traces_df.apply(
            lambda x: SPIKE_GADGETS_MULTIPLIER * (x[col] - np.median(x[col])) / x[MAD_column], axis=1)

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_RMS".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: (x / np.sqrt(np.mean(x ** 2))).astype(np.float32))

    zscore_columns = [col for col in lfp_traces_df.columns if "zscore" in col]
    for col in zscore_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_mask".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: np.abs(x) >= zscore_threshold)

    for col in original_trace_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_trace_filtered".format(brain_region)
        mask_column = "{}_lfp_mask".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df.apply(
            lambda x: helper_update_array_by_mask(
                x[col], x[mask_column]), axis=1)

    filtered_trace_column = [
        col for col in lfp_traces_df if "lfp_trace_filtered" in col]
    for col in filtered_trace_column:
        brain_region = col.split("_")[0]
        updated_column = "{}_lfp_RMS_filtered".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: (x / np.sqrt(np.nanmean(x ** 2))).astype(np.float32))
    #print("done preprocessing")
    return lfp_traces_df


def calculate_power(
        lfp_traces_df,
        resample_rate,
        time_halfbandwidth_product,
        time_window_duration,
        time_window_step):
    """
    Calculates the power of the LFP traces using the multitaper method.
    Args:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces.
        resample_rate (int): The resample rate for the LFP traces.
        time_halfbandwidth_product (float): The time halfbandwidth product for the multitaper method.
        time_window_duration (float): The duration of the time window for the multitaper method.
        time_window_step (float): The step size for the time window for the multitaper method.
    Returns:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces with the power calculated.
        - the power columns have [time sample x freq]
    """
    #print("calculating power")
    input_columns = [
        col for col in lfp_traces_df.columns if "trace" in col or "RMS" in col]

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
                lambda x: Connectivity.from_multitaper(x))

            lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                lambda x: x.frequencies)
            lfp_traces_df[power_col] = lfp_traces_df[connectivity_col].apply(
                lambda x: x.power().squeeze()
            )

            lfp_traces_df[power_col] = lfp_traces_df[power_col].apply(
                lambda x: x.astype(np.float16))

            lfp_traces_df = lfp_traces_df.drop(
                columns=[
                    multitaper_col,
                    connectivity_col],
                errors="ignore")

        except Exception as e:
            print(e)
            

    lfp_traces_df["power_timestamps"] = lfp_traces_df["lfp_timestamps"].apply(
        lambda x: x[(resample_rate // 2):(-resample_rate // 2):(resample_rate // 2)])
    lfp_traces_df["power_calculation_frequencies"] = lfp_traces_df[[
        col for col in lfp_traces_df.columns if "power_frequencies" in col][0]].copy()
    lfp_traces_df = lfp_traces_df.drop(
        columns=[
            col for col in lfp_traces_df.columns if "power_frequencies" in col],
        errors="ignore")

    return lfp_traces_df

def calculate_phase(lfp_traces_df, fs):
    """
    Calculates the phase of the LFP traces using the Hilbert transform.
    Args:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces.
        fs (int): The (down) sampling rate for the LFP traces.
    Returns:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces with the phase calculated.
    """
    #print("calculating phase")
    from scipy.signal import butter, filtfilt, hilbert

    order = 4
    RMS_columns = [
        col for col in lfp_traces_df if "RMS" in col and "filtered" in col and "all" not in col]

    # Filter for theta band
    freq_band = [4, 12]
    b, a = butter(order, freq_band, fs=fs, btype='band')
    for col in RMS_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_theta_band".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: filtfilt(b, a, x, padtype=None).astype(np.float32))

    # Filter for gamma band
    freq_band = [30, 50]
    b, a = butter(order, freq_band, fs=fs, btype='band')
    for col in RMS_columns:
        brain_region = col.split("_")[0]
        updated_column = "{}_gamma_band".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: filtfilt(b, a, x, padtype=None).astype(np.float32))

    # Calculate phase
    band_columns = [col for col in lfp_traces_df if "band" in col]
    for col in band_columns:
        brain_region = col.replace("_band", "")
        updated_column = "{}_phase".format(brain_region)
        lfp_traces_df[updated_column] = lfp_traces_df[col].apply(
            lambda x: np.angle(hilbert(x), deg=False))

    return lfp_traces_df


def calculate_coherence(
        lfp_traces_df,
        resample_rate,
        time_halfbandwidth_product,
        time_window_duration,
        time_window_step):
    """
    Calculates the coherence of the LFP traces using the multitaper method.
    Args:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces.
        resample_rate (int): The resample rate for the LFP traces.
        time_halfbandwidth_product (float): The time halfbandwidth product for the multitaper method.
        time_window_duration (float): The duration of the time window for the multitaper method.
        time_window_step (float): The step size for the time window for the multitaper method.
    Returns:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces with the coherence calculated.
    """
    #print("calculating coherence")
    input_columns = [
        col for col in lfp_traces_df.columns if "trace" in col or "RMS" in col]
    all_suffixes = set(["_".join(col.split("_")[1:]) for col in input_columns])
    brain_region_pairs = helper_generate_pairs(
        list(set([col.split("lfp")[0].strip("_") for col in input_columns])))

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
                    lambda x: Connectivity.from_multitaper(x))

                lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.frequencies)
                lfp_traces_df[coherence_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.coherence_magnitude()[:, :, 0, 1])

                lfp_traces_df[coherence_col] = lfp_traces_df[coherence_col].apply(
                    lambda x: x.astype(np.float16))

            except Exception as e:
                print(e)
                
            lfp_traces_df = lfp_traces_df.drop(
                columns=[
                    multitaper_col,
                    connectivity_col],
                errors="ignore")

    lfp_traces_df["coherence_timestamps"] = lfp_traces_df["lfp_timestamps"].apply(
        lambda x: x[(resample_rate // 2):(-resample_rate // 2):(resample_rate // 2)])
    lfp_traces_df["coherence_calculation_frequencies"] = lfp_traces_df[[
        col for col in lfp_traces_df.columns if "coherence_frequencies" in col][0]].copy()
    lfp_traces_df = lfp_traces_df.drop(
        columns=[
            col for col in lfp_traces_df.columns if "coherence_frequencies" in col],
        errors="ignore")

    return lfp_traces_df


def calculate_granger_causality(
        lfp_traces_df,
        resample_rate,
        time_halfbandwidth_product,
        time_window_duration,
        time_window_step):
    """
    Calculates the Granger causality of the LFP traces using the multitaper method.
    Args:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces.
        resample_rate (int): The resample rate for the LFP traces.
        time_halfbandwidth_product (float): The time halfbandwidth product for the multitaper method.
        time_window_duration (float): The duration of the time window for the multitaper method.
        time_window_step (float): The step size for the time window for the multitaper method.
    Returns:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces with the Granger causality calculated.
    """
    #print("calculating granger causality")
    input_columns = [
        col for col in lfp_traces_df.columns if "trace" in col or "RMS" in col]
    all_suffixes = set(["_".join(col.split("_")[1:]) for col in input_columns])
    brain_region_pairs = helper_generate_pairs(
        list(set([col.split("lfp")[0].strip("_") for col in input_columns])))

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
                    lambda x: Connectivity.from_multitaper(x))

                lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.frequencies)

                lfp_traces_df[granger_1_2_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.pairwise_spectral_granger_prediction()[:, :, 0, 1])

                lfp_traces_df[granger_2_1_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.pairwise_spectral_granger_prediction()[:, :, 1, 0])

                lfp_traces_df[granger_1_2_col] = lfp_traces_df[granger_1_2_col].apply(
                    lambda x: x.astype(np.float16))
                lfp_traces_df[granger_2_1_col] = lfp_traces_df[granger_2_1_col].apply(
                    lambda x: x.astype(np.float16))

            except Exception as e:
                print(e)
              

            lfp_traces_df = lfp_traces_df.drop(
                columns=[
                    multitaper_col,
                    connectivity_col],
                errors="ignore")

    lfp_traces_df["granger_timestamps"] = lfp_traces_df["lfp_timestamps"].apply(
        lambda x: x[(resample_rate // 2):(-resample_rate // 2):(resample_rate // 2)])
    lfp_traces_df["granger_calculation_frequencies"] = lfp_traces_df[[
        col for col in lfp_traces_df.columns if "granger_frequencies" in col][0]].copy()
    lfp_traces_df = lfp_traces_df.drop(
        columns=[
            col for col in lfp_traces_df.columns if "granger_frequencies" in col],
        errors="ignore")

    return lfp_traces_df


# %%
TIME_HALFBANDWIDTH_PRODUCT = 2
TIME_WINDOW_DURATION = 1
TIME_WINDOW_STEP = 0.5
RESAMPLE_RATE=1000
CHANNEL_MAPPING_DF = load_channel_map(r"D:\social_ephys_pilot2_cum\data\tester\channel_mapping.xlsx")

def do_the_thing(input_dir):
    #input_dir = r"D:\social_ephys_pilot2_cum\data\phase2\20230803_101331_1.1_1t1bL_FCN.rec"
    #output_path = r"D:\social_ephys_pilot2_cum\data\phase2\20230803_101331_1.1_1t1bL_FCN.rec\outputs"
    output_path = os.path.join(input_dir, 'outputs')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    lfpobject = LfpRecordingObject(input_dir, output_path)
    metadata_path = os.path.join(output_path, 'metadata.pkl')
    metadata_df = unpickle_this(metadata_path)
    recording_names_dict = extract_lfp_traces(
            ALL_SESSION_DIR=input_dir,
            ECU_STREAM_ID="ECU",
            TRODES_STREAM_ID="trodes",
            RECORDING_EXTENTION="*merged.rec",
            #you do not want the .rec not merged file 
            LFP_FREQ_MIN=0.5,
            LFP_FREQ_MAX=300,
            ELECTRIC_NOISE_FREQ=60,
            LFP_SAMPLING_RATE=1000,
            EPHYS_SAMPLING_RATE=20000)

    recording_names_dict

    lfp_df = combine_lfp_traces_and_metadata(
            SPIKEGADGETS_EXTRACTED_DF =metadata_df,
            recording_name_to_all_ch_lfp = recording_names_dict,
            CHANNEL_MAPPING_DF = CHANNEL_MAPPING_DF,
            EPHYS_SAMPLING_RATE = 20000,
            LFP_SAMPLING_RATE = 1000,
            LFP_RESAMPLE_RATIO=20,
            ALL_CH_LFP_COL="all_ch_lfp",
            SUBJECT_COL="Subject",
            CURRENT_SUBJECT_COL="current_subject")

    lfp_preprocessed_traces_df = preprocess_lfp_data(lfp_df, voltage_scaling_value=0.195, zscore_threshold=4, resample_rate=1000)
    lfp_power_df = calculate_power(lfp_preprocessed_traces_df, 1000, 2, 1, 0.5)
    lfp_phase_df = calculate_phase(lfp_power_df, fs = 1000)

    lfp_coherence_df = calculate_coherence(lfp_phase_df,
                                        RESAMPLE_RATE,
                                        TIME_HALFBANDWIDTH_PRODUCT,
                                        TIME_WINDOW_DURATION,
                                        TIME_WINDOW_STEP)

    lfp_granger_df = calculate_granger_causality(lfp_coherence_df,
                                                RESAMPLE_RATE,
                                                TIME_HALFBANDWIDTH_PRODUCT,
                                                TIME_WINDOW_DURATION,
                                                TIME_WINDOW_STEP)

    # %%
    pickle_dir = os.path.join(output_path, 'lfp_all_df.pkl')
    pickle_this(lfp_granger_df, pickle_dir)
    print("DONE")

if __name__ == "__main__":
    p1 = Process(target=do_the_thing, args=(r"D:\social_ephys_pilot2_cum\data\phase2\20230818_115728_1.1_1t1bL_NFC.rec",))
    p1.start()
    # = Process(target=do_the_thing, args=(r"D:\social_ephys_pilot2_cum\data\phase2\20230817_113746_1.2_2t2bL_FCN.rec",))
    #p2.start()
    p1.join()
    #p2.join()
    
    

# %%
