import glob
import subprocess
import os
from collections import defaultdict
import trodes.read_exported
import pandas as pd
import numpy as np

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
                # print("Timestamp Array for {}: ".format(video_basename))
                # print(timestamp_array)

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

    # Rename columns during t

    trodes_metadata_df = trodes_metadata_df.reset_index()
    print(type(trodes_metadata_df))
    print("HERE")
    print(trodes_metadata_df.columns)
    trodes_metadata_df = trodes_metadata_df.rename(columns={'level_0': 'session_dir', 'level_1': 'recording', 'level_2': 'metadata_dir', 'level_3': 'metadata_file'}, errors="ignore")
    trodes_metadata_df["session_path"] = trodes_metadata_df["session_dir"].map(session_to_path)
    print("After filter")
    print(trodes_metadata_df.columns)
    return trodes_metadata_df

def add_subjects_to_metadata(metadata):
    # TODO: find a better way to do this without regex on the session_dir
    metadata["all_subjects"] = metadata["session_dir"].apply(
        lambda x: x.replace("-", "_").split("subj")[-1].split("t")[0].strip("_").replace("_", ".").split(".and."))
    metadata["all_subjects"] = metadata["all_subjects"].apply(
        lambda x: sorted([i.strip().strip(".") for i in x]))
    metadata["current_subject"] = metadata["recording"].apply(
        lambda x: x.replace("-", "_").split("subj")[-1].split("t")[0].strip("_").replace("_", ".").split(".and.")[0])
    return metadata

def get_trodes_video_df(trodes_metadata_df):
    trodes_video_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"] == "video_timestamps"].copy().reset_index(
        drop=True)
    trodes_video_df = trodes_video_df[trodes_video_df["metadata_file"] == "1"].copy()
    trodes_video_df["video_timestamps"] = trodes_video_df["first_item_data"]
    trodes_video_df = trodes_video_df[["filename", "video_timestamps", "session_dir"]].copy()
    trodes_video_df = trodes_video_df.rename(columns={"filename": "video_name"})
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
    return trodes_state_df

def adjust_first_timestamps(trodes_metadata_df):
    trodes_metadata_df["first_dtype_name"] = trodes_metadata_df["data"].apply(lambda x: x.dtype.names[0])
    trodes_metadata_df["first_item_data"] = trodes_metadata_df["data"].apply(lambda x: x[x.dtype.names[0]])
    trodes_metadata_df["last_dtype_name"] = trodes_metadata_df["data"].apply(lambda x: x.dtype.names[-1])
    trodes_metadata_df["last_item_data"] = trodes_metadata_df["data"].apply(lambda x: x[x.dtype.names[-1]])

    trodes_metadata_df = add_subjects_to_metadata(trodes_metadata_df)

    metadata_filtered = metadata[metadata["metadata_file"].isin(["raw", "DIO", "video_timestamps"])]
    trodes_raw_df = metadata_filtered[
        (metadata_filtered["metadata_dir"] == "raw") & (metadata_filtered["metadata_file"] == "timestamps")].copy()
    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(lambda x: x[0])

    recording_to_first_timestamp = trodes_raw_df.set_index('session_dir')['first_timestamp'].to_dict()
    trodes_metadata_df["first_timestamp"] = trodes_metadata_df["session_dir"].map(recording_to_first_timestamp)

    trodes_state_df = get_trodes_state_df(trodes_metadata_df)

    trodes_video_df = get_trodes_video_df(trodes_metadata_df)

    trodes_state_df = pd.merge(trodes_state_df, trodes_video_df, on=["session_dir"], how="inner")
    trodes_state_df["event_frames"] = trodes_state_df.apply(
        lambda x: find_nearest_indices(x["event_timestamps"], x["video_timestamps"]), axis=1)
    state_cols_to_keep = ["session_dir", "metadata_file", "event_timestamps", "event_frames", "video_name"]
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

    return trodes_metadata_df, trodes_state_df, trodes_video_df


input_dir = "/Volumes/chaitra/reward_competition_extension/data/standard/2023_06_*/*.rec"
output_dir = "/Volumes/chaitra/test_lfp/proc/"
TONE_DIN = "dio_ECU_Din1"
TONE_STATE = 1
experiment_dir = "/Volumes/chaitra/reward_competition_extension/data"
#convert_to_mp4(experiment_dir)
paths = {}
session_to_trodes_temp, paths= extract_all_trodes(input_dir)
session_to_trodes_temp = add_video_timestamps(session_to_trodes_temp, input_dir)
metadata = create_metadata_df(session_to_trodes_temp, paths)
metadata, state_df, video_df = adjust_first_timestamps(metadata)