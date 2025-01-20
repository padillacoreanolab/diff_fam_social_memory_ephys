import numpy as np


def threshold_bouts(start_stop_array, min_iti, min_bout):
    """
    thresholds behavior bouts
    by combining behavior bouts with interbout intervals of < min_iti
    and then removing remaining bouts of < min_bout

    Args (3 total):
        start_stop_array: numpy array of dim (# of bouts, 2)
        min_iti: float, min interbout interval in seconds
        min_bout: float, min bout length in seconds

    Returns (1):
        start_stop_array: numpy array (ndim=(n bouts, 2))
            of start&stop times (ms)
    """

    start_stop_array = np.sort(start_stop_array.flatten())
    times_to_delete = []
    if min_iti > 0:
        for i in range(1, len(start_stop_array) - 1, 2):
            if (start_stop_array[i + 1] - start_stop_array[i]) < min_iti:
                times_to_delete.extend([i, i + 1])
    start_stop_array = np.delete(start_stop_array, times_to_delete)
    bouts_to_delete = []
    if min_bout > 0:
        for i in range(0, len(start_stop_array) - 1, 2):
            if start_stop_array[i + 1] - start_stop_array[i] < min_bout:
                bouts_to_delete.extend([i, i + 1])
    start_stop_array = np.delete(start_stop_array, bouts_to_delete)
    no_bouts = len(start_stop_array) / 2
    start_stop_array = np.reshape(start_stop_array, (int(no_bouts), 2))

    return start_stop_array


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
        start_stop_arrays.append(threshold_bouts(start_stop_array, min_iti, min_bout))
    start_stop_array = np.concatenate(start_stop_arrays)
    organizer = np.argsort(start_stop_array[:, 0])
    start_stop_array = start_stop_array[organizer]

    return start_stop_array * 1000


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
