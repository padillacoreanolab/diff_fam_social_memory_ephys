import numpy as np
import behavior.behavioral_epoch_tools as bet


def get_behavior_bouts(boris_df, subject, behavior, min_iti=0, min_bout=0):
    """
    extracts behavior bout start and stop times from a boris df
    thresholds individually by subject and behavior
    returns start_stop_array ordered by start values

    Args (5 total, 3 required):
        boris_df: pandas dataframe of a boris file (aggregated event table)
        subject: list of strings or ints, desired subject(s)
                as written in boris_df, i.e. 'novel' or 1.1
        behavior: list of strings, desired behavior(s) (as written in boris_df)
        min_iti: float, default=0, bouts w/ itis(s) < min_iti will be combined
        min_bout: float, default=0, bouts < min_bout(s) will be deleted

    Returns (1):
        numpy array (ndim=(n bouts, 2)) of start&stop times (ms)
    """
    start_stop_arrays = []
    for mouse in subject:
        subject_df = boris_df[boris_df["Subject"] == mouse]
        behavior_arrays = []
        for act in behavior:
            behavior_df = subject_df[subject_df["Behavior"] == act]
            start_stop_array = behavior_df[["Start (s)", "Stop (s)"]].to_numpy()
            behavior_arrays.append(start_stop_array)
        start_stop_array = np.concatenate(behavior_arrays)
        start_stop_arrays.append(bet.threshold_bouts(start_stop_array, min_iti, min_bout))
    start_stop_array = np.concatenate(start_stop_arrays)
    organizer = np.argsort(start_stop_array[:, 0])
    start_stop_array = start_stop_array[organizer]
    start_stop_array_ms = start_stop_array * 1000  # convert to ms

    return start_stop_array_ms


def get_behavior_bouts_frame(boris_df, cameratimestamps, first_timestamp, subject, behavior, min_iti=0, min_bout=0):
    """
    extracts behavior bout start and stop times from a boris df from a NONE-MAC scored video and thresholds behavior
    bouts individually by subject and behavior returns start_stop_array ordered by start values. takes in a
    cameratimestamps array and first_timestamp to adjust for the start of the recording (cameratimestamps second 0
    starts at stream) so adjusts start and stop values using first_timestamp so the 0th second aligns to play.

    Args (6 total, 4 required):
        boris_df: pandas dataframe of a boris file (aggregated event table)
        cameratimestamps: numpy array of camera timestamps (in seconds) read from .videoTimeStamps file generated from
            trodes
        first_timestamp: int, first timestamp of the recording in timestamps from trodes
        subject: list of strings or ints, desired subject(s)
                as written in boris_df, i.e. 'novel' or 1.1
        behavior: list of strings, desired behavior(s) (as written in boris_df)
        min_iti: float, default=0, bouts w/ itis(s) < min_iti will be combined
        min_bout: float, default=0, bouts < min_bout(s) will be deleted

    Returns (1):
        numpy array (ndim=(n bouts, 2)) of start&stop times (ms)
    """
    start_stop_arrays = []
    for mouse in subject:
        subject_df = boris_df[boris_df["Subject"] == mouse]
        subject_df.loc[:, "Image index stop"] = subject_df["Image index stop"].fillna(
            subject_df["Image index start"].shift(-1)
        )
        behavior_arrays = []
        for act in behavior:
            behavior_df = subject_df[subject_df["Behavior"] == act]
            start_stop_array = behavior_df[["Image index start", "Image index stop"]].to_numpy()
            behavior_arrays.append(start_stop_array)
        start_stop_array = np.concatenate(behavior_arrays)
        start_stop_array = start_stop_array.astype(int)
        start_stop_array_s = cameratimestamps[start_stop_array]
        start_stop_arrays.append(bet.threshold_bouts(start_stop_array_s, min_iti, min_bout))
    if start_stop_arrays:
        # Filter out any empty arrays (extra safety)
        valid_arrays = [arr for arr in start_stop_arrays if arr.size > 0]

        if valid_arrays:
            start_stop_array = np.concatenate(valid_arrays)
            organizer = np.argsort(start_stop_array[:, 0])
            start_stop_array_stream_indexed = start_stop_array[organizer]
            start_stop_array_play_indexed = start_stop_array_stream_indexed - (first_timestamp / 20000)
            # adjust for first timestamp
            start_stop_array_ms = start_stop_array_play_indexed * 1000  # convert to ms
            return start_stop_array_ms

    # Return empty array with correct shape if no data found
    return np.empty((0, 2))


def get_behavior_bouts_fps(boris_df, cameratimestamps, first_timestamp, subject, behavior, min_iti=0, min_bout=0):
    """
    extracts behavior bout start and stop times from a boris df from a MAC scored video and thresholds behavior
    bouts individually by subject and behavior returns start_stop_array ordered by start values. takes in a
    cameratimestamps array and first_timestamp to adjust for the start of the recording (cameratimestamps second 0
    starts at stream) so adjusts start and stop values using first_timestamp so the 0th second aligns to play.

    Uses the FPS of the video to convert start and stop times (s) from the corresponding columns to frame such that
    (Start (s) X FPS) rounded to neared integer.

    Args (6 total, 4 required):
        boris_df: pandas dataframe of a boris file (aggregated event table)
        cameratimestamps: numpy array of camera timestamps (in seconds) read from .videoTimeStamps file generated from
            trodes
        first_timestamp: int, first timestamp of the recording in timestamps from trodes
        subject: list of strings or ints, desired subject(s)
                as written in boris_df, i.e. 'novel' or 1.1
        behavior: list of strings, desired behavior(s) (as written in boris_df)
        min_iti: float, default=0, bouts w/ itis(s) < min_iti will be combined
        min_bout: float, default=0, bouts < min_bout(s) will be deleted

    Returns (1):
        numpy array (ndim=(n bouts, 2)) of start&stop times (ms)
    """
    start_stop_arrays = []
    for mouse in subject:
        subject_df = boris_df[boris_df["Subject"] == mouse]
        if not subject_df.empty:
            behavior_arrays = []
            column_names = subject_df.columns
            for column_name in column_names:
                if "FPS" in column_name:
                    fps_column = column_name
                    break
            try:
                fps = int(subject_df[fps_column].unique()[0])
            except IndexError:
                print(subject_df)
            for act in behavior:
                behavior_df = subject_df[subject_df["Behavior"] == act]
                start_stop_array = behavior_df[["Start (s)", "Stop (s)"]].to_numpy()
                behavior_arrays.append(start_stop_array)
            start_stop_array = np.concatenate(behavior_arrays)
            start_stop_array = np.round(start_stop_array).astype(int) * fps
            if start_stop_array[-1, 1] > len(cameratimestamps):
                if start_stop_array[-1, 1] - len(cameratimestamps) == 1:
                    print("last stop: ", start_stop_array[-1, 1])
                    print("number of timestamps: ", len(cameratimestamps))
                    start_stop_array[-1, 1] = len(cameratimestamps) - 1
            start_stop_array_s = cameratimestamps[start_stop_array]
            start_stop_arrays.append(bet.threshold_bouts(start_stop_array_s, min_iti, min_bout))
        else:
            start_stop_array = np.array([])
            start_stop_arrays.append(start_stop_array)
    if start_stop_arrays:
        # Filter out any empty arrays (extra safety)
        valid_arrays = [arr for arr in start_stop_arrays if arr.size > 0]

        if valid_arrays:
            start_stop_array = np.concatenate(valid_arrays)
            organizer = np.argsort(start_stop_array[:, 0])
            start_stop_array_stream_indexed = start_stop_array[organizer]
            start_stop_array_play_indexed = start_stop_array_stream_indexed - (first_timestamp / 20000)
            # adjust for first timestamp
            start_stop_array_ms = start_stop_array_play_indexed * 1000  # convert to ms
            return start_stop_array_ms

    # Return empty array with correct shape if no data found
    return np.empty((0, 2))


def save_behavior_bouts(directory, boris_df, subject, behavior, min_iti=0, min_bout=0, filename=None):
    """
    saves a numpy array of start&stop times (ms)
    as filename: subject_behavior_bouts.npy

    Args (7 total, 4 required):
        directory: path to folder where filename.npy will be saved
            path format: './folder/folder/'
        boris_df: pandas dataframe of a boris file (aggregated event table)
        subject: list of strings, desired subjects (as written in boris_df)
        behavior: list of strings, desired behaviors (as written in boris_df)
        min_iti: float, default=0, bouts w/ itis(s) < min_iti will be combined
        min_bout: float, default=0, bouts < min_bouts(s) will be deleted
        filename: string, default=None, must end in .npy

    Returns:
        none
    """
    bouts_array = get_behavior_bouts(boris_df, subject, behavior, min_iti, min_bout)
    if filename is None:
        if type(subject) is list:
            subject = "_".join(subject)
        if type(behavior) is list:
            behavior = "_".join(behavior)
        subject = subject.replace(" ", "")
        behavior = behavior.replace(" ", "")
        filename = f"{subject}_{behavior}_bouts.npy"

    np.save(directory + filename, bouts_array)


def reciprocal_bouts(bouts_array, time_window=None):
    """
    takes a numpy array of start&stop times (ms)
    and returns a numpy array of reciprocal bouts
    i.e. bout 1: start1, stop1, start2, stop2

    Args (3 total, 1 required):
        bouts_array: numpy array (ndim=(n bouts, 2)) of start&stop times (ms)
        time_window: float, default=None, time window in s converted to ms. If None,
            there will be no time window considered and all overlapping bouts will be
            returned as reciprocal bouts. If time_window not None, only bouts that are
            overlapping and within the time_window will be returned as reciprocal bouts.

    Returns (1):
        numpy array of arrays of start&stop times (ms)
    """
    if time_window is None:
        time_window = 0

    return None
