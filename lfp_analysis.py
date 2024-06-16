import glob
import subprocess
import os
import warnings
from collections import defaultdict
import trodes.read_exported
import pandas as pd
import numpy as np
from scipy import stats
from spectral_connectivity import Multitaper, Connectivity
import logging
import h5py
from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp


class LfpExperiment:
    """
    list_of_lfp_objects: list of LFPObject objects
    for file in recording_dir:
        generate_lfp_object(file)

    mainly for analysis, graphs, and overall trends
    """

    def __init__(self, experiment):
        self.list_of_lfp_objects = {}
        self.experiment = experiment

    def add_lfp_object(self, lfp_object, subject):
        self.list_of_lfp_objects[(subject, self.experiment)] = lfp_object


class LfpRecordingObject:
    def make_object(self):
        # call extract_all_trodes
        session_to_trodes_temp, paths = extract_all_trodes(input_dir=self.path)
        # call add_video_timestamps
        session_to_trodes_temp = add_video_timestamps(
            session_to_trodes_data=session_to_trodes_temp, directory_path=self.path)
        # call create_metadata_df
        metadata = create_metadata_df(
            session_to_trodes=session_to_trodes_temp, session_to_path=paths)
        # call adjust_first_timestamps
        self.metadata, self.state_df, self.video_df, self.final_df = adjust_first_timestamps(
            trodes_metadata_df=metadata, output_dir=self.path, experiment_prefix=self.experiment_name)

        print(self.output_path)

        # Pickle all
        self.metadata.to_pickle(self.output_path + "/metadata.pkl")
        self.state_df.to_pickle(self.output_path + "/state_df.pkl")
        self.video_df.to_pickle(self.output_path + "/video_df.pkl")
        self.final_df.to_pickle(self.output_path + "/final_df.pkl")

        print("LFP Object has been created for " +
              self.subject + " at " + self.path)

    def make_power_df(self):
        LFP_TRACES_DF = preprocess_lfp_data(
            lfp_traces_df=self.spike_df,
            voltage_scaling_value=self.VOLTAGE_SCALING_VALUE,
            zscore_threshold=self.zscore_threshold,
            resample_rate=self.RESAMPLE_RATE)
        self.LFP_TRACES_DF = LFP_TRACES_DF
        print("LFP TRACES DF")
        print(LFP_TRACES_DF.head())
        # call get_power
        power_df = calculate_power(
            self.spike_df,
            self.RESAMPLE_RATE,
            self.TIME_HALFBANDWIDTH_PRODUCT,
            self.TIME_WINDOW_DURATION,
            self.TIME_WINDOW_STEP)
        # assign variables
        self.power_df = power_df

    def make_phase_df(self):
        # call get_phase
        phase_df = calculate_phase(self.spike_df, fs=1000)
        # assign variables
        self.phase_df = phase_df

    def make_coherence_df(self):
        # call get_coherence
        # lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step
        coherence_df = calculate_coherence(self.spike_df, self.RESAMPLE_RATE,
                                           self.TIME_HALFBANDWIDTH_PRODUCT,
                                           self.TIME_WINDOW_DURATION,
                                           self.TIME_WINDOW_STEP)
        # assign variables
        self.coherence_df = coherence_df

    def make_granger_df(self):
        # call get_granger
        # lfp_traces_df, resample_rate, time_halfbandwidth_product, time_window_duration, time_window_step
        granger_df = calculate_granger_causality(lfp_traces_df=self.spike_df, resample_rate=self.RESAMPLE_RATE,
                                                 time_halfbandwidth_product=self.TIME_HALFBANDWIDTH_PRODUCT,
                                                 time_window_duration=self.TIME_WINDOW_DURATION,
                                                 time_window_step=self.TIME_WINDOW_STEP)
        # assign variables
        self.granger_df = granger_df

    def make_filter_bands_df(self):
        # call get_filter_bands
        # (lfp_spectral_df, theta_band, gamma_band, output_dir, output_prefix):
        filter_bands_df = calculate_filter_bands(
            lfp_spectral_df=self.power_df,
            theta_band=self.BAND_TO_FREQ["theta"],
            gamma_band=self.BAND_TO_FREQ["gamma"],
            output_dir=os.getcwd(),
            output_prefix="test")
        # assign variables
        self.filter_bands_df = filter_bands_df

    def make_sleap_df(self):
        # call get_start_stop
        """
        sleap_dir,
                       output_dir,
                       med_pc_width,
                       med_pc_height,
                       frame_rate,
                       window_size,
                       distance_threshold,
                       start_stop_frame_df,
                       lfp_spectral_df,
                       thorax_index,
                       output_prefix
        """
        # todo: init with medpc width and height; window size; distance threshold; thorax index
        sleap_df, start_stop_df = process_sleap_data(sleap_dir=self.sleap_path,
                                                     output_dir=self.output_path,
                                                     med_pc_width=1,
                                                     med_pc_height=1,
                                                     frame_rate=self.frame_rate,
                                                     window_size=90,
                                                     distance_threshold=0.1,
                                                     start_stop_frame_df=pd.read_excel(
                                                         self.events_path),
                                                     lfp_spectral_df=self.filter_bands_df,
                                                     thorax_index=1,
                                                     output_prefix=self.experiment_name)
        # assign variables
        self.sleap_df = sleap_df
        self.start_stop_df = start_stop_df

    def analyze_sleap(self):
        # todo: thorax index hardcoded
        thorax_index = 1
        analyze_sleap_file(start_stop_frame_df=self.start_stop_df, plot_output_dir=self.output_path + "/plots/",
                           output_prefix=self.experiment_name,
                           thorax_index=thorax_index, thorax_plots=True, save_plots=True)

    def add_spike_times(self):
        # takes lfp spectral and phy dir
        # lfp_spectral_df, grouped_df, all_spike_time_df

        list_phy_files = glob.glob(os.path.join(self.phy_curation_path, "*"))

        self.power_df, self.grouped_df, self.all_spike_time_df = add_spike_to_phy(
            lfp_spectral_df=self.filter_bands_df,
            phy_curation_path=list_phy_files,
            output_dir=self.output_path,
            output_prefix=self.experiment_name,
            sampling_rate=self.sampling_rate)

    def make_output_dir(self):
        print("IN MAKE OUTPUT DIR")
        os.makedirs(self.output_path, exist_ok=True)
        self.output_path = self.output_path + "/" + self.subject
        os.makedirs(self.output_path, exist_ok=True)
        print("Output path is " + self.output_path)

    def add_labels(self):
        self.labels_and_spectral = make_labels_df(
            labels_df=self.labels_df, filter_bands_df=self.filter_bands_df)

    def add_label_encoding(self):
        self.labels_and_spectral = encode_labels(filter_bands_df=self.filter_bands_df,
                                                 labels_df=self.labels_df,
                                                 encoding_dict=self.encoding_dict)

    def __init__(self,
                 path,
                 channel_map_path,
                 sleap_path,
                 events_path,
                 phy_curation_path,
                 labels_path,
                 experiment_name,
                 subject,
                 output_path,
                 encoding_dict,
                 ecu=False,
                 sampling_rate=20000,
                 frame_rate=22):
        self.path = path
        self.channel_map_path = channel_map_path
        self.sleap_path = sleap_path
        self.events_path = events_path
        self.phy_curation_path = phy_curation_path
        self.labels_df = pd.read_excel(labels_path)
        self.experiment_name = experiment_name
        self.events = {}
        self.channel_map = {}
        self.recording = None
        self.subject = subject
        self.sampling_rate = sampling_rate
        self.frame_rate = frame_rate
        self.encoding_dict = encoding_dict

        self.output_path = output_path

        # add variables from make object function
        self.metadata = None
        self.state_df = None
        self.video_df = None
        self.final_df = None
        self.pkl_path = None

        # inputs needed for notebook 2 (power)
        self.LFP_TRACES_DF = None
        self.original_trace_columns = None

        # TODO: hard coding these variables for power
        self.VOLTAGE_SCALING_VALUE = 0.195
        self.zscore_threshold = 4
        self.RESAMPLE_RATE = 1000
        self.TIME_HALFBANDWIDTH_PRODUCT = 2
        self.TIME_WINDOW_DURATION = 1
        self.TIME_WINDOW_STEP = 0.5
        self.BAND_TO_FREQ = {"theta": (4, 12), "gamma": (30, 51)}

        # granger
        self.power_df = None
        self.phase_df = None
        self.coherence_df = None
        self.granger_df = None

        # notebook 3 "bands"
        self.filter_bands_df = None

        # notebook 4 sleap, events
        self.sleap_df = None
        self.start_stop_df = None

        # notebook 5
        self.spike_times_df = None

        self.all_spike_time_df = None
        self.grouped_df = None

        # labels notebook
        self.labels_and_spectral = None

        self.make_output_dir()
        self.make_object()

        # get channel map and lfp
        # ALL_SESSION_DIR, ECU_STREAM_ID, TRODES_STREAM_ID,
        # RECORDING_EXTENTION, LFP_FREQ_MIN, LFP_FREQ_MAX, ELECTRIC_NOISE_FREQ,
        # LFP_SAMPLING_RATE, EPHYS_SAMPLING_RATE):
        self.recording_names_dict = extract_lfp_traces(
            ALL_SESSION_DIR=self.path,
            ECU_STREAM_ID="ECU",
            TRODES_STREAM_ID="trodes",
            RECORDING_EXTENTION="*.rec",
            LFP_FREQ_MIN=0.5,
            LFP_FREQ_MAX=300,
            ELECTRIC_NOISE_FREQ=60,
            LFP_SAMPLING_RATE=1000,
            EPHYS_SAMPLING_RATE=20000)

        self.channel_map = load_channel_map(
            channel_map_path=self.channel_map_path)

        self.spike_df = combine_lfp_traces_and_metadata(
            SPIKEGADGETS_EXTRACTED_DF=self.final_df,
            recording_name_to_all_ch_lfp=self.recording_names_dict,
            CHANNEL_MAPPING_DF=self.channel_map,
            CURRENT_SUBJECT_COL="current_subject",
            SUBJECT_COL="Subject",
            ALL_CH_LFP_COL="all_ch_lfp",
            LFP_RESAMPLE_RATIO=20,
            EPHYS_SAMPLING_RATE=20000,
            LFP_SAMPLING_RATE=1000)

        # temporarily pickle the spike_df for debugging
        self.spike_df.to_pickle(self.output_path + "/spike_df.pkl")

        self.make_power_df()
        print("Power dataframe has been created at " +
              self.output_path + "/power_df.pkl")
        self.power_df.to_pickle(self.output_path + "/power_df.pkl")

        self.make_phase_df()
        print("Phase dataframe has been created at " +
              self.output_path + "/phase_df.pkl")
        self.phase_df.to_pickle(self.output_path + "/phase_df.pkl")

        self.make_coherence_df()
        print("Coherence dataframe has been created at " +
              self.output_path + "/coherence_df.pkl")
        self.coherence_df.to_pickle(self.output_path + "/coherence_df.pkl")

        self.make_granger_df()
        print("Granger dataframe has been created at " +
              self.output_path + "/granger_df.pkl")
        self.granger_df.to_pickle(self.output_path + "/granger_df.pkl")

        self.make_filter_bands_df()
        print("Filter bands dataframe has been created at " +
              self.output_path + "/filter_bands_df.pkl")
        self.filter_bands_df.to_pickle(
            self.output_path + "/filter_bands_df.pkl")

        self.make_sleap_df()
        print("Sleap and start/stop dataframes has been created at " + self.output_path + "/sleap_df.pkl and "
              + self.output_path + "/start_stop_df.pkl")
        self.sleap_df.to_pickle(self.output_path + "/sleap_df.pkl")
        self.start_stop_df.to_pickle(self.output_path + "/start_stop_df.pkl")

        self.analyze_sleap()
        print("Analysis of sleap data has been completed")

        self.add_spike_times()
        print("Spike times have been added to the phy file at " +
              self.output_path + "/spike_times_df.pkl")
        self.spike_times_df.to_pickle(self.output_path + "/spike_times_df.pkl")
        self.grouped_df.to_pickle(self.output_path + "/grouped_df.pkl")
        self.all_spike_time_df.to_pickle(
            self.output_path + "/all_spike_time_df.pkl")

        # label notebook functions
        self.add_labels()
        self.labels_and_spectral.to_pickle(
            self.output_path + "/labels_and_spectral.pkl")

        # TODO: export notebook functions
        #self.add_label_encoding()


def helper_filter_array_by_values(arr, start_value=0, stop_value=1000000):
    """
    Filters elements of a 1D or rows of a 2D numpy array based on specified value range.

    Parameters:
    - arr (numpy array): The input numpy array to filter.
    - start_value (numeric): The lower bound for the filtering. Default is 0.
    - stop_value (numeric): The upper bound for the filtering. Default is 1000000.

    Returns:
    - numpy array: A numpy array containing only the filtered elements or rows.

    Raises:
    - ValueError: If the input array has more than two dimensions.
    """
    result = np.array(arr)

    if result.ndim == 1:
        # Apply filter for a 1D array
        mask = (result > start_value) & (result < stop_value)
        return result[mask], mask

    elif result.ndim == 2:
        # Apply filter based on the first column for a 2D array
        mask = (result[:, 0] > start_value) & (result[:, 0] < stop_value)
        return result[mask], mask

    else:
        raise ValueError("The input array must be either 1D or 2D.")


def helper_combine_grouped_rows(df, array_columns):
    """
    Combine rows within groups of a DataFrame. Uses the `overlay_arrays` for specified columns
    and takes the first instance for other columns.

    Parameters:
    - df (pd.DataFrame): DataFrame to process.
    - array_columns (list): List of column names in `df` that contain array values to be combined using `overlay_arrays`.

    Returns:
    - pd.DataFrame: DataFrame after combining rows within groups.
    """
    def custom_aggregator(x):
        if x.name in array_columns:
            # Reduce the column by overlaying arrays
            return x.dropna().aggregate(lambda arrays: arrays.reduce(helper_overlay_arrays))
        else:
            # For other columns, simply return the first element
            return x.iloc[0]

    # Apply the custom aggregator to each column individually
    return df.groupby(df.index).aggregate(custom_aggregator)


def helper_extract_start_stop_elements(array, start_index=0, stop_index=-1):
    """
    Extracts elements from an array at specified start and stop indices.

    Parameters:
    - array (list or array-like): The array from which elements are to be extracted.
    - start_index (int): The index of the start element. Default is 0.
    - stop_index (int): The index of the stop element. Default is -1, which corresponds to the last element.

    Returns:
    - tuple: A tuple containing the elements at the start and stop indices.

    If the stop_index is out of the array's range, it defaults to the last element of the array.
    """
    if stop_index >= len(array) or stop_index < 0:
        stop_index = -1  # Ensure the stop_index points to the last element if it's out of range
    return array[start_index], array[stop_index]


def helper_overlay_arrays(array1, array2):
    """
    Overlays two 2D NumPy arrays of the same shape, preferring non-NaN values from the first array.
    If both arrays have a non-NaN value at the same position, the value from the first array is used.

    Parameters:
    - array1 (np.ndarray): The primary 2D array.
    - array2 (np.ndarray): The secondary 2D array, used only where array1 has NaNs.

    Returns:
    - np.ndarray: A 2D array composed of the overlaid results of array1 and array2.

    Raises:
    - ValueError: If `array1` and `array2` do not have the same shape.
    """
    if array1.shape != array2.shape:
        raise ValueError("Both arrays must have the same shape.")

    # Create a copy of the first array to ensure that no changes are made to the original
    result = np.copy(array1)

    # Find indices where array1 is NaN and array2 is not NaN
    mask = np.isnan(array1) & ~np.isnan(array2)

    # Place non-NaN values from array2 where array1 has NaNs
    result[mask] = array2[mask]

    return result


def helper_mask_slices(array_2d, slice_index):
    """
    Masks elements outside a specified slice in a 2D array with NaN.

    Parameters:
    - array_2d (np.ndarray): The input 2D array to mask.
    - slice_index (tuple): A tuple of two integers that specifies the start and end indices
                           of the slice to retain. The elements outside this slice are set to NaN.

    Returns:
    - np.ndarray: A 2D array with elements outside the specified slice set to NaN.

    Raises:
    - ValueError: If `slice_index` is not a tuple or does not contain exactly two elements.
    - IndexError: If the slice indices are out of the array bounds.
    """
    if not isinstance(slice_index, tuple) or len(slice_index) != 2:
        raise ValueError(
            "slice_index must be a tuple of two integers (start, end).")

    try:
        result = array_2d.copy()
        mask_2d = np.ones(result.shape, dtype=bool)
        mask_2d[slice_index[0]:slice_index[1]] = False

        # Set values outside the defined slice to NaN
        result[mask_2d] = np.nan
        return result

    except IndexError as e:
        warnings.warn(f"Slice index out of bounds: {e}")
        return np.nan
    except Exception as e:
        warnings.warn(f"An error occurred while masking the array: {e}")
        return np.nan


def helper_filter_by_timestamp_range(start, stop, timestamps, items):
    """
    Filters an array of timestamps and corresponding items based on a timestamp range.

    Parameters:
    - start (int or float): The start of the timestamp range.
    - stop (int or float): The end of the timestamp range.
    - timestamps (numpy.ndarray): A sorted array of timestamps.
    - items (numpy.ndarray): An array of items corresponding to the timestamps.

    Returns:
    - tuple: Two numpy.ndarrays, the filtered timestamps and the corresponding items.
    """
    # Create a boolean mask for the timestamps within the range
    mask = (timestamps >= start) & (timestamps <= stop)

    # Apply the mask to the timestamps
    filtered_timestamps = timestamps[mask]

    # Apply the mask to the items, adjusting the length if necessary
    if len(items) > len(mask):
        # If items is longer than mask, shorten items
        filtered_items = items[:len(mask)][mask]
    elif len(items) < len(mask):
        # If items is shorter than mask, pad items with NaNs
        padded_items = np.pad(
            items, (0, len(mask) - len(items)), constant_values=np.nan)
        filtered_items = padded_items[mask]
    else:
        # If items and mask are the same length, just apply the mask
        filtered_items = items[mask]

    return filtered_timestamps, filtered_items


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


def helper_compute_velocity(node_loc, window_size=25, polynomial_order=3):
    """
    Calculate the velocity of tracked nodes from pose data.

    The function utilizes the Savitzky-Golay filter to smooth the data and compute the velocity.

    Parameters:
    ----------
    node_loc : numpy.ndarray
        The location of nodes, represented as an array of shape [frames, 2].
        Each row represents x and y coordinates for a particular frame.

    window_size : int, optional
        The size of the window used for the Savitzky-Golay filter.
        Represents the number of consecutive data points used when smoothing the data.
        Default is 25.

    polynomial_order : int, optional
        The order of the polynomial fit to the data within the Savitzky-Golay filter window.
        Default is 3.

    Returns:
    -------
    numpy.ndarray
        The velocity for each frame, calculated from the smoothed x and y coordinates.

    """
    node_loc_vel = np.zeros_like(node_loc)

    # For each coordinate (x and y), smooth the data and calculate the
    # derivative (velocity)
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(
            node_loc[:, c], window_size, polynomial_order, deriv=1)

    # Calculate the magnitude of the velocity vectors for each frame
    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def get_sleap_tracks_from_h5(filename):
    """
    Retrieve pose tracking data (tracks) from a SLEAP-generated h5 file.

    This function is intended for use with Pandas' apply method on columns containing filenames.

    Parameters:
    ----------
    filename : str
        Path to the SLEAP h5 file containing pose tracking data.

    Returns:
    -------
    np.ndarray
        A transposed version of the 'tracks' dataset in the provided h5 file.

    Example:
    --------
    df['tracks'] = df['filename_column'].apply(get_sleap_tracks_from_h5)

    """
    with h5py.File(filename, "r") as f:
        return f["tracks"][:].T


def get_node_names_from_sleap(filename):
    """
    Retrieve node names from a SLEAP h5 file.

    Parameters:
    - filename (str): Path to the SLEAP h5 file.

    Returns:
    - list of str: List of node names.
    """
    with h5py.File(filename, "r") as f:
        return [n.decode() for n in f["node_names"][:]]


def fill_missing(Y, kind="linear"):
    """Fills missing values independently along each dimension after the first."""

    # Store initial shape.
    initial_shape = Y.shape

    # Flatten after first dim.
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate along each slice.
    for i in range(Y.shape[-1]):
        y = Y[:, i]

        # Build interpolant.
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)

        # Fill missing
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)

        # Fill leading or trailing NaNs with the nearest non-NaN values
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(
            mask), np.flatnonzero(~mask), y[~mask])

        # Save slice
        Y[:, i] = y

    # Restore to initial shape.
    Y = Y.reshape(initial_shape)

    return Y


def compute_velocity(node_loc, window_size=25, polynomial_order=3):
    """
    Calculate the velocity of tracked nodes from pose data.

    The function utilizes the Savitzky-Golay filter to smooth the data and compute the velocity.

    Parameters:
    ----------
    node_loc : numpy.ndarray
        The location of nodes, represented as an array of shape [frames, 2].
        Each row represents x and y coordinates for a particular frame.

    window_size : int, optional
        The size of the window used for the Savitzky-Golay filter.
        Represents the number of consecutive data points used when smoothing the data.
        Default is 25.

    polynomial_order : int, optional
        The order of the polynomial fit to the data within the Savitzky-Golay filter window.
        Default is 3.

    Returns:
    -------
    numpy.ndarray
        The velocity for each frame, calculated from the smoothed x and y coordinates.

    """
    node_loc_vel = np.zeros_like(node_loc)

    # For each coordinate (x and y), smooth the data and calculate the
    # derivative (velocity)
    for c in range(node_loc.shape[-1]):
        node_loc_vel[:, c] = savgol_filter(
            node_loc[:, c], window_size, polynomial_order, deriv=1)

    # Calculate the magnitude of the velocity vectors for each frame
    node_vel = np.linalg.norm(node_loc_vel, axis=1)

    return node_vel


def extract_sleap_data(filename):
    """
    Extracts coordinates, names of body parts, and track names from a SLEAP file.

    Parameters:
    - filename (str): Path to the SLEAP file.

    Returns:
    - tuple: A tuple containing the following elements:
        * locations (numpy.ndarray): Array containing the coordinates.
        * node_names (list of str): List of body part names.
        * track_names (list of str): List of track names.

    Example: locations, node_names, track_names = extract_sleap_data("path/to/sleap/file.h5")
    """
    result = {}
    with h5py.File(filename, "r") as f:
        result["locations"] = f["tracks"][:].T
        result["node_names"] = [n.decode() for n in f["node_names"][:]]
        result["track_names"] = [n.decode() for n in f["track_names"][:]]

    return result


def rescale_dimension_in_array(arr, dimension=0, ratio=1):
    """
    Rescale values of a specified dimension in a 3D numpy array for the entire array.

    Parameters:
    - arr (numpy.ndarray): A 3D numpy array where the third dimension is being rescaled.
    - dimension (int, default=0): Specifies which dimension (0 or 1) of the third
                                  dimension in the array should be rescaled.
                                  For instance, in many contexts:
                                  0 represents the x-coordinate,
                                  1 represents the y-coordinate.
    - ratio (float, default=1): The scaling factor to be applied.

    Returns:
    - numpy.ndarray: The rescaled array.
    """

    arr[:, :, dimension] *= ratio
    return arr


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
            session_to_trodes_data[session_basename] = trodes.read_exported.organize_all_trodes_export(
                session)
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

            for video_timestamps in glob.glob(
                    os.path.join(session, "*cameraHWSync")):
                video_basename = os.path.basename(video_timestamps)
                print("Current Video Name: {}".format(video_basename))
                timestamp_array = trodes.read_exported.read_trodes_extracted_data_file(
                    video_timestamps)

                if "video_timestamps" not in session_to_trodes_data[session_basename][session_basename]:
                    session_to_trodes_data[session_basename][session_basename]["video_timestamps"] = defaultdict(
                        dict)

                session_to_trodes_data[session_basename][session_basename]["video_timestamps"][video_basename.split(
                    ".")[-3]] = timestamp_array
                print("Timestamp Array for {}: ".format(video_basename))
                print(session_to_trodes_data[session_basename][session_basename]
                      ["video_timestamps"][video_basename.split(".")[-3]])

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

    print("unique recordings ")
    print(trodes_metadata_df["recording"].unique())
    return trodes_metadata_df


def add_subjects_to_metadata(metadata):
    """
    Adds the subjects to the metadata dataframe.
    Args:
        metadata (pandas dataframe): Generated from create_metadata_df.
    Returns:
        metadata (pandas dataframe): A dataframe containing the metadata for each session with the subjects added.
    """
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
    """
    Extracts the video data from the trodes_metadata_df and calculates the first timestamp for each session.
    Args:
        trodes_metadata_df (pandas dataframe): Generated from create_metadata_df.
    Returns:
        trodes_video_df (pandas dataframe): A dataframe containing the video data for each session.
    """
    trodes_video_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"] == "video_timestamps"].copy().reset_index(
        drop=True)
    trodes_video_df = trodes_video_df[trodes_video_df["metadata_file"] == "1"].copy(
    )
    trodes_video_df["video_timestamps"] = trodes_video_df["first_item_data"]
    trodes_video_df = trodes_video_df[[
        "filename", "video_timestamps", "session_dir"]].copy()
    trodes_video_df = trodes_video_df.rename(
        columns={"filename": "video_name"})
    print(trodes_video_df.head())
    return trodes_video_df


def get_trodes_state_df(trodes_metadata_df):
    """
    Extracts the state data from the trodes_metadata_df and calculates the timestamps for each event.
    Args:
        trodes_metadata_df (pandas dataframe): Generated from create_metadata_df.
    Returns:
        trodes_state_df (pandas dataframe): A dataframe containing the state data for each session.
    """
    trodes_state_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"].isin([
        "DIO"])].copy()
    trodes_state_df = trodes_metadata_df[trodes_metadata_df["id"].isin(
        ["ECU_Din1", "ECU_Din2"])].copy()
    trodes_state_df["event_indexes"] = trodes_state_df.apply(
        lambda x: np.column_stack([np.where(x["last_item_data"] == 1)[
            0], np.where(x["last_item_data"] == 1)[0] + 1]),
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
    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(
        lambda x: x[0])
    trodes_raw_cols = ['session_dir', 'recording', 'original_file', 'session_path', 'current_subject',
                       'first_item_data',
                       'first_timestamp', 'all_subjects']
    trodes_raw_df = trodes_raw_df[trodes_raw_cols].reset_index(
        drop=True).copy()
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
    trodes_final_df = pd.merge(
        trodes_raw_df,
        trodes_state_df,
        on=["session_dir"],
        how="inner")
    trodes_final_df = trodes_final_df.rename(
        columns={"first_item_data": "raw_timestamps"})
    trodes_final_df = trodes_final_df.drop(
        columns=["metadata_file"], errors="ignore")
    trodes_final_df = trodes_final_df.sort_values(
        ["session_dir", "recording"]).reset_index(drop=True).copy()
    sorted_columns = sorted(trodes_final_df.columns,
                            key=lambda x: x.split("_")[-1])
    trodes_final_df = trodes_final_df[sorted_columns].copy()
    for col in [col for col in trodes_final_df.columns if "timestamps" in col]:
        trodes_final_df[col] = trodes_final_df.apply(lambda x: x[col].astype(np.int32) - np.int32(x["first_timestamp"]),
                                                     axis=1)

    for col in [col for col in trodes_final_df.columns if "frames" in col]:
        trodes_final_df[col] = trodes_final_df[col].apply(
            lambda x: x.astype(np.int32))

    print("trodes final df")
    trodes_final_df["last_timestamp"] = trodes_final_df["raw_timestamps"].apply(
        lambda x: x[-1])
    print(trodes_final_df.head())
    print(trodes_final_df.columns)
    return trodes_final_df


def merge_state_video_df(trodes_state_df, trodes_video_df):
    """
    Cleans the trodes_state_df and trodes_video_df dataframes and merges them on the session_dir column.
    Args:
        trodes_state_df (pandas dataframe): Generated from get_trodes_state_df.
        trodes_video_df (pandas dataframe): Generated from get_trodes_video_df.
    Returns:
        trodes_state_df (pandas dataframe): A dataframe containing the state data for each session.
    """
    trodes_state_df = pd.merge(
        trodes_state_df,
        trodes_video_df,
        on=["session_dir"],
        how="inner")
    trodes_state_df["event_frames"] = trodes_state_df.apply(
        lambda x: helper_find_nearest_indices(x["event_timestamps"], x["video_timestamps"]), axis=1)
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

    trodes_state_df["tone_timestamps"] = trodes_state_df["event_timestamps"].apply(
        lambda x: x[0])
    trodes_state_df["port_entry_timestamps"] = trodes_state_df["event_timestamps"].apply(
        lambda x: x[1])

    trodes_state_df["tone_frames"] = trodes_state_df["event_frames"].apply(
        lambda x: x[0])
    trodes_state_df["port_entry_frames"] = trodes_state_df["event_frames"].apply(
        lambda x: x[1])
    trodes_state_df = trodes_state_df.drop(
        columns=[
            "event_timestamps",
            "event_frames"],
        errors="ignore")
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
    trodes_metadata_df = trodes_metadata_df[trodes_metadata_df["metadata_dir"].isin(
        metadata_cols_to_keep)].copy()
    trodes_metadata_df = trodes_metadata_df[~trodes_metadata_df["metadata_file"].str.contains(
        "out")]
    trodes_metadata_df = trodes_metadata_df[~trodes_metadata_df["metadata_file"].str.contains(
        "coordinates")]
    trodes_metadata_df = trodes_metadata_df.reset_index(drop=True)

    trodes_raw_df = trodes_metadata_df[
        (trodes_metadata_df["metadata_dir"] == "raw") & (trodes_metadata_df["metadata_file"] == "timestamps")].copy()

    trodes_raw_df["first_timestamp"] = trodes_raw_df["first_item_data"].apply(
        lambda x: x[0])

    recording_to_first_timestamp = trodes_raw_df.set_index(
        'session_dir')['first_timestamp'].to_dict()
    print(recording_to_first_timestamp)
    trodes_metadata_df["first_timestamp"] = trodes_metadata_df["session_dir"].map(
        recording_to_first_timestamp)
    print(trodes_metadata_df["first_timestamp"])

    trodes_state_df = get_trodes_state_df(trodes_metadata_df)

    trodes_video_df = get_trodes_video_df(trodes_metadata_df)

    trodes_raw_df = get_trodes_raw_df(trodes_metadata_df)

    trodes_state_df = merge_state_video_df(trodes_state_df, trodes_video_df)

    trodes_final_df = make_final_df(
        trodes_raw_df, trodes_state_df, trodes_video_df)

    return trodes_metadata_df, trodes_state_df, trodes_video_df, trodes_final_df


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


def extract_lfp_traces(ALL_SESSION_DIR, ECU_STREAM_ID, TRODES_STREAM_ID, RECORDING_EXTENTION,
                       LFP_FREQ_MIN, LFP_FREQ_MAX, ELECTRIC_NOISE_FREQ, LFP_SAMPLING_RATE, EPHYS_SAMPLING_RATE):
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
    print("ALL SESSION DIR is " + ALL_SESSION_DIR)
    for session_dir in glob.glob(ALL_SESSION_DIR):
        for recording_path in glob.glob(
                os.path.join(session_dir, RECORDING_EXTENTION)):
            try:
                recording_basename = os.path.splitext(
                    os.path.basename(recording_path))[0]
                current_recording = se.read_spikegadgets(
                    recording_path, stream_id=ECU_STREAM_ID)
                current_recording = se.read_spikegadgets(
                    recording_path, stream_id=TRODES_STREAM_ID)
                print(recording_basename)

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
    print("LENGTH OF RECORDING NAME TO ALL CH LFP")
    print(len(recording_name_to_all_ch_lfp))
    return recording_name_to_all_ch_lfp


def combine_lfp_traces_and_metadata(SPIKEGADGETS_EXTRACTED_DF, recording_name_to_all_ch_lfp, CHANNEL_MAPPING_DF,
                                    EPHYS_SAMPLING_RATE,
                                    LFP_SAMPLING_RATE, LFP_RESAMPLE_RATIO=20, ALL_CH_LFP_COL="all_ch_lfp",
                                    SUBJECT_COL="Subject", CURRENT_SUBJECT_COL="current_subject"):
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
    print("recording name to all channel")
    print(recording_name_to_all_ch_lfp)
    print(SPIKEGADGETS_EXTRACTED_DF.columns)
    lfp_trace_condition = (
        SPIKEGADGETS_EXTRACTED_DF["recording"].isin(recording_name_to_all_ch_lfp))
    print(lfp_trace_condition)

    SPIKEGADGETS_LFP_DF = SPIKEGADGETS_EXTRACTED_DF[lfp_trace_condition].copy(
    ).reset_index(drop=True)
    print("on line 494")
    SPIKEGADGETS_LFP_DF["all_ch_lfp"] = SPIKEGADGETS_LFP_DF["recording"].map(
        recording_name_to_all_ch_lfp)
    print("on line 496")
    SPIKEGADGETS_LFP_DF["LFP_timestamps"] = SPIKEGADGETS_LFP_DF.apply(
        lambda row: np.arange(0, row["all_ch_lfp"].get_total_samples() * LFP_RESAMPLE_RATIO, LFP_RESAMPLE_RATIO,
                              dtype=int), axis=1)
    print("on line 500")
    SPIKEGADGETS_LFP_DF = pd.merge(
        SPIKEGADGETS_LFP_DF,
        CHANNEL_MAPPING_DF,
        left_on=CURRENT_SUBJECT_COL,
        right_on=SUBJECT_COL,
        how="left")
    print("on line 502")
    SPIKEGADGETS_LFP_DF["all_channels"] = SPIKEGADGETS_LFP_DF["all_ch_lfp"].apply(
        lambda x: x.get_channel_ids())
    SPIKEGADGETS_LFP_DF["region_channels"] = SPIKEGADGETS_LFP_DF[["spike_interface_mPFC",
                                                                  "spike_interface_vHPC",
                                                                  "spike_interface_BLA",
                                                                  "spike_interface_LH",
                                                                  "spike_interface_MD"]].to_dict('records')
    SPIKEGADGETS_LFP_DF["region_channels"] = SPIKEGADGETS_LFP_DF["region_channels"].apply(
        lambda x: sorted(x.items(), key=lambda item: int(item[1])))
    print(SPIKEGADGETS_LFP_DF["region_channels"].iloc[0])
    print("on line 506")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
    print(SPIKEGADGETS_LFP_DF.head())
    print(SPIKEGADGETS_LFP_DF.columns)

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

    # print number of rows in SPIKEGADGETS_LFP_DF
    print("num rows in spike df")
    print(len(SPIKEGADGETS_LFP_DF))
    # Apply the modified function
    SPIKEGADGETS_LFP_DF["all_region_lfp_trace"] = SPIKEGADGETS_LFP_DF.apply(
        get_traces_with_progress, axis=1)
    print("on line 508")
    SPIKEGADGETS_LFP_DF["per_region_lfp_trace"] = SPIKEGADGETS_LFP_DF.apply(lambda row: dict(zip(["{}_lfp_trace".format(
        t[0].strip("spike_interface_")) for t in row["region_channels"]], row["all_region_lfp_trace"])), axis=1)
    SPIKEGADGETS_FINAL_DF = pd.concat([SPIKEGADGETS_LFP_DF.copy(
    ), SPIKEGADGETS_LFP_DF['per_region_lfp_trace'].apply(pd.Series).copy()], axis=1)
    print("on line 510")
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

    print("done combining lfp traces and metadata")

    return SPIKEGADGETS_FINAL_DF


### START OF NOTEBOOK 2 ###


def preprocess_lfp_data(lfp_traces_df, voltage_scaling_value,
                        zscore_threshold, resample_rate):
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
    print("beginning preprocessing")
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
    print("done preprocessing")
    return lfp_traces_df


def calculate_power(lfp_traces_df, resample_rate,
                    time_halfbandwidth_product, time_window_duration, time_window_step):
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
    """
    print("calculating power")
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
                lambda x: Connectivity.from_multitaper(x)
            )

            lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                lambda x: x.frequencies
            )
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
        fs (int): The sampling rate for the LFP traces.
    Returns:
        lfp_traces_df (pandas dataframe): A dataframe containing the LFP traces with the phase calculated.
    """
    print("calculating phase")
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


def calculate_coherence(lfp_traces_df, resample_rate,
                        time_halfbandwidth_product, time_window_duration, time_window_step):
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
    print("calculating coherence")
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
                    lambda x: Connectivity.from_multitaper(x)
                )

                lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.frequencies
                )
                lfp_traces_df[coherence_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.coherence_magnitude()[:, :, 0, 1]
                )

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


def calculate_granger_causality(lfp_traces_df, resample_rate,
                                time_halfbandwidth_product, time_window_duration, time_window_step):
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
    print("calculating granger causality")
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
                    lambda x: Connectivity.from_multitaper(x)
                )

                lfp_traces_df[frequencies_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.frequencies
                )

                lfp_traces_df[granger_1_2_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.pairwise_spectral_granger_prediction()[
                        :, :, 0, 1]
                )

                lfp_traces_df[granger_2_1_col] = lfp_traces_df[connectivity_col].apply(
                    lambda x: x.pairwise_spectral_granger_prediction()[
                        :, :, 1, 0]
                )

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


### START OF NOTEBOOK 3 ###


def calculate_filter_bands(lfp_spectral_df, theta_band,
                           gamma_band, output_dir, output_prefix):
    """
    Filters the LFP spectral data for theta and gamma bands, and saves the result to a file.

    Args:
        lfp_spectral_df (pd.DataFrame): LFP spectral data.
        theta_band (tuple): Frequency range for theta band.
        gamma_band (tuple): Frequency range for gamma band.
        output_dir (str): Directory where data is saved.
        output_prefix (str): Prefix for the output files.
    Returns:
        None
    """
    print("In filter bands function")
    # Filter theta/gamma for power
    print(lfp_spectral_df.columns)
    power_columns = [col for col in lfp_spectral_df.columns if
                     "power" in col and "calculation" not in col and "time" not in col]
    for col in power_columns:
        print(col)
        brain_region_name = col.split("power")[0].strip("_")
        theta_power_col = f"{brain_region_name}_power_theta"
        gamma_power_col = f"{brain_region_name}_power_gamma"
        lfp_spectral_df[theta_power_col] = lfp_spectral_df.apply(lambda x: np.nanmean(x[col][:, (x[
            "power_calculation_frequencies"] >=
            theta_band[0]) & (x[
                "power_calculation_frequencies"] <=
            theta_band[
                1])],
            axis=1), axis=1)
        lfp_spectral_df[gamma_power_col] = lfp_spectral_df.apply(lambda x: np.nanmean(x[col][:, (x[
            "power_calculation_frequencies"] >=
            gamma_band[0]) & (x[
                "power_calculation_frequencies"] <=
            gamma_band[
                1])],
            axis=1), axis=1)

    # Filter theta/gamma for coherence
    coherence_columns = [col for col in lfp_spectral_df.columns if
                         "coherence" in col and "calculation" not in col and "time" not in col]
    for col in coherence_columns:
        brain_region_name = "_".join(col.split("_")[:2])
        theta_coherence_col = f"{brain_region_name}_coherence_theta"
        gamma_coherence_col = f"{brain_region_name}_coherence_gamma"
        lfp_spectral_df[theta_coherence_col] = lfp_spectral_df.apply(lambda x: np.nanmean(x[col][:, (x[
            "coherence_calculation_frequencies"] >=
            theta_band[0]) & (
            x[
                "coherence_calculation_frequencies"] <=
            theta_band[
                1])],
            axis=1), axis=1)
        lfp_spectral_df[gamma_coherence_col] = lfp_spectral_df.apply(lambda x: np.nanmean(x[col][:, (x[
            "coherence_calculation_frequencies"] >=
            gamma_band[0]) & (
            x[
                "coherence_calculation_frequencies"] <=
            gamma_band[
                1])],
            axis=1), axis=1)

    # Filter theta/gamma for granger
    granger_columns = [col for col in lfp_spectral_df.columns if
                       "granger" in col and "calculation" not in col and "time" not in col]
    for col in granger_columns:
        print(col)
        brain_region_name = "-to-".join(col.split("_")[:2])
        theta_granger_col = f"{brain_region_name}_granger_theta"
        gamma_granger_col = f"{brain_region_name}_granger_gamma"
        lfp_spectral_df[theta_granger_col] = lfp_spectral_df.apply(lambda x: np.nanmean(x[col][:, (x[
            "granger_calculation_frequencies"] >=
            theta_band[0]) & (x[
                "granger_calculation_frequencies"] <=
            theta_band[
                1])],
            axis=1), axis=1)
        lfp_spectral_df[gamma_granger_col] = lfp_spectral_df.apply(lambda x: np.nanmean(x[col][:, (x[
            "granger_calculation_frequencies"] >=
            gamma_band[0]) & (x[
                "granger_calculation_frequencies"] <=
            gamma_band[
                1])],
            axis=1), axis=1)

    return lfp_spectral_df


### START OF NOTEBOOK 4 ##


def convert_pixels_to_cm(start_stop_frame_df, med_pc_width, med_pc_height):
    """
    Converts the pixel values in the input dataframe to centimeters.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pixel values.
        med_pc_width (float): The width of the video in pixels.
        med_pc_height (float): The height of the video in pixels.
    Returns:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pixel values converted to centimeters.
    """
    print("converting pixels to cm")
    print(start_stop_frame_df.columns)
    start_stop_frame_df["bottom_width"] = start_stop_frame_df["corner_to_coordinate"].apply(
        lambda x: x["box_bottom_right"][0] - x["box_bottom_left"][0])
    start_stop_frame_df["top_width"] = start_stop_frame_df["corner_to_coordinate"].apply(
        lambda x: x["box_top_right"][0] - x["box_top_left"][0])
    start_stop_frame_df["right_height"] = start_stop_frame_df["corner_to_coordinate"].apply(
        lambda x: x["box_bottom_right"][1] - x["box_top_right"][1])
    start_stop_frame_df["left_height"] = start_stop_frame_df["corner_to_coordinate"].apply(
        lambda x: x["box_bottom_left"][1] - x["box_top_left"][1])
    start_stop_frame_df["average_height"] = start_stop_frame_df.apply(
        lambda row: (row["right_height"] + row["left_height"]) / 2, axis=1)
    start_stop_frame_df["average_width"] = start_stop_frame_df.apply(
        lambda row: (row["bottom_width"] + row["top_width"]) / 2, axis=1)
    start_stop_frame_df["width_ratio"] = med_pc_width / \
        start_stop_frame_df["average_width"]
    start_stop_frame_df["height_ratio"] = med_pc_height / \
        start_stop_frame_df["average_height"]

    start_stop_frame_df["in_video_subjects"] = start_stop_frame_df["in_video_subjects"].apply(
        lambda x: x.split("_"))
    start_stop_frame_df["subject_to_tracks"] = start_stop_frame_df.apply(
        lambda x: {
            k: v for k,
            v in x["subject_to_tracks"].items() if k in x["in_video_subjects"]},
        axis=1)
    start_stop_frame_df["rescaled_locations"] = start_stop_frame_df.apply(
        lambda x: {
            key: fill_missing(
                rescale_dimension_in_array(
                    value,
                    dimension=0,
                    ratio=x["width_ratio"])) for key,
            value in x["subject_to_tracks"].items()},
        axis=1)
    start_stop_frame_df["rescaled_locations"] = start_stop_frame_df.apply(
        lambda x: {
            key: rescale_dimension_in_array(
                value,
                dimension=1,
                ratio=x["height_ratio"]) for key,
            value in x["rescaled_locations"].items()},
        axis=1)

    normalized = pd.json_normalize(start_stop_frame_df["corner_to_coordinate"])
    start_stop_frame_df = pd.concat([start_stop_frame_df.drop(
        ["corner_to_coordinate"], axis=1), normalized], axis=1)

    for corner in start_stop_frame_df["corner_parts"].iloc[0]:
        start_stop_frame_df[corner] = start_stop_frame_df.apply(
            lambda x: [
                x[corner][0] *
                x["width_ratio"],
                x[corner][1] *
                x["height_ratio"]],
            axis=1)

    return start_stop_frame_df


def create_individual_pose_tracking_columns(start_stop_frame_df):
    """
    Creates columns for the individual pose tracking data.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data.
    Returns:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data with the individual
        columns created.
    """
    start_stop_frame_df = start_stop_frame_df.dropna(subset="current_subject")
    start_stop_frame_df["agent"] = start_stop_frame_df.apply(lambda x: list(
        (set(x["tracked_subject"]) - set([x["current_subject"]]))), axis=1)
    start_stop_frame_df["agent"] = start_stop_frame_df["agent"].apply(
        lambda x: x[0] if len(x) == 1 else None)
    start_stop_frame_df["subject_locations"] = start_stop_frame_df.apply(
        lambda x: x["rescaled_locations"][x["current_subject"]], axis=1)
    start_stop_frame_df["agent_locations"] = start_stop_frame_df.apply(
        lambda x: x["rescaled_locations"].get(
            x["agent"], np.nan) if x["agent"] else np.nan, axis=1)
    # start_stop_frame_df = start_stop_frame_df.drop(["sleap_glob", "subject_to_index", "subject_to_tracks", "corner_parts", "corner_to_coordinate", "bottom_width", "top_width", "right_height", "left_height", "average_height", "average_width", "width_ratio", "height_ratio", 'locations', 'track_names', 'sleap_path', 'corner_path', 'all_sleap_data', 'rescaled_locations'], errors="ignore", axis=1)

    return start_stop_frame_df


def calculate_velocity(start_stop_frame_df, window_size,
                       frame_rate, thorax_index):
    """
    Calculates the velocity of the subject and agent thorax.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data.
        window_size (int): The window size for the velocity calculation.
        frame_rate (int): The frame rate of the video.
        thorax_index (int): The index of the thorax in the body parts list.
    Returns:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data with the velocity
        calculated.
    """
    start_stop_frame_df["body_parts"].apply(lambda x: x.index("thorax"))
    start_stop_frame_df["subject_thorax_velocity"] = start_stop_frame_df.apply(lambda x: helper_compute_velocity(
        x["subject_locations"][:, x["body_parts"].index("thorax"), :], window_size=frame_rate) * frame_rate, axis=1)
    start_stop_frame_df["subject_thorax_velocity"] = start_stop_frame_df["subject_thorax_velocity"].apply(
        lambda x: x.astype(np.float16) if x is not np.nan else np.nan)
    start_stop_frame_df["agent_thorax_velocity"] = start_stop_frame_df.apply(
        lambda x: helper_compute_velocity(
            x["agent_locations"][
                :,
                x["body_parts"].index("thorax"),
                :],
            window_size=frame_rate) *
        frame_rate if x["agent_locations"] is not np.nan else np.nan,
        axis=1)
    start_stop_frame_df["agent_thorax_velocity"] = start_stop_frame_df["agent_thorax_velocity"].apply(
        lambda x: x.astype(np.float16) if x is not np.nan else np.nan)

    return start_stop_frame_df


def calculate_distance_to_reward_port(start_stop_frame_df, thorax_index):
    """
    Calculates the distance between the subject and agent thorax to the reward port.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data.
        thorax_index (int): The index of the thorax in the body parts list.
    Returns:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data with the distance
    """
    start_stop_frame_df["subject_thorax_to_reward_port"] = start_stop_frame_df.apply(lambda x: np.linalg.norm(
        x["subject_locations"][:, x["body_parts"].index("thorax"), :] - x["reward_port"], axis=1), axis=1)
    start_stop_frame_df["subject_thorax_to_reward_port"] = start_stop_frame_df["subject_thorax_to_reward_port"].apply(
        lambda x: x.astype(np.float16) if x is not np.nan else np.nan)
    start_stop_frame_df["agent_thorax_to_reward_port"] = start_stop_frame_df.apply(lambda x: np.linalg.norm(
        x["agent_locations"][:, x["body_parts"].index("thorax"), :] - x["reward_port"], axis=1) if x[
        "agent_locations"] is not np.nan else np.nan,
        axis=1)
    start_stop_frame_df["agent_thorax_to_reward_port"] = start_stop_frame_df["agent_thorax_to_reward_port"].apply(
        lambda x: x.astype(np.float16) if x is not np.nan else np.nan)

    return start_stop_frame_df


def process_sleap_tracks(start_stop_frame_df, sleap_dir,
                         med_pc_width, med_pc_height):
    """
    Processes the SLEAP tracks.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data.
        sleap_dir (str): The directory containing the SLEAP data.
        med_pc_width (float): The width of the video in pixels.
        med_pc_height (float): The height of the video in pixels.
    Returns:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data with the SLEAP
        tracks processed.
    """
    print(start_stop_frame_df.columns)
    start_stop_frame_df["tracked_subject"] = start_stop_frame_df["tracked_subject"].apply(
        lambda x: str(x).split("_"))
    start_stop_frame_df["current_subject"] = start_stop_frame_df["tracked_subject"]
    start_stop_frame_df = start_stop_frame_df.explode("current_subject")
    start_stop_frame_df["sleap_glob"] = start_stop_frame_df["sleap_name"].apply(
        lambda x: glob.glob(os.path.join(sleap_dir, "**", x)))
    start_stop_frame_df = start_stop_frame_df[start_stop_frame_df['sleap_glob'].apply(
        lambda x: len(x) >= 1)]
    start_stop_frame_df = start_stop_frame_df.reset_index(drop=True)
    start_stop_frame_df["sleap_path"] = start_stop_frame_df["sleap_glob"].apply(
        lambda x: x[0])
    start_stop_frame_df["all_sleap_data"] = start_stop_frame_df["sleap_path"].apply(
        lambda x: extract_sleap_data(x))
    start_stop_frame_df["body_parts"] = start_stop_frame_df["sleap_path"].apply(
        lambda x: get_node_names_from_sleap(x))
    start_stop_frame_df["locations"] = start_stop_frame_df["all_sleap_data"].apply(
        lambda x: x["locations"])
    start_stop_frame_df["track_names"] = start_stop_frame_df["all_sleap_data"].apply(
        lambda x: x["track_names"])

    print(start_stop_frame_df["track_names"])
    print(start_stop_frame_df["track_names"].dtype)
    print(start_stop_frame_df["tracked_subject"])
    print(start_stop_frame_df["tracked_subject"].dtype)
    print(start_stop_frame_df.columns)
    print(start_stop_frame_df.head())

    # Getting the indexes of each subject from the track list
    start_stop_frame_df["subject_to_index"] = start_stop_frame_df.apply(
        lambda x: {k: x["track_names"].index(k) for k in x["tracked_subject"] if k in x["track_names"]}, axis=1)
    start_stop_frame_df["subject_to_tracks"] = start_stop_frame_df.apply(
        lambda x: {k: v for k, v in x["subject_to_index"].items()}, axis=1)
    start_stop_frame_df["subject_to_tracks"] = start_stop_frame_df.apply(
        lambda x: {k: x["locations"][:, :, :, v] for k, v in x["subject_to_index"].items()}, axis=1)

    start_stop_frame_df["corner_path"] = start_stop_frame_df["sleap_path"].apply(
        lambda x: x.replace(
            "id_corrected.h5",
            "corner.h5").replace(
            ".fixed",
            "").replace(
            ".round_1",
            "").replace(
            ".1_subj",
            "").replace(
            ".2_subj",
            ""))
    start_stop_frame_df["corner_parts"] = start_stop_frame_df["corner_path"].apply(
        lambda x: get_node_names_from_sleap(x))
    start_stop_frame_df = start_stop_frame_df[start_stop_frame_df["corner_parts"].apply(
        lambda x: "reward_port" in x)]
    start_stop_frame_df["corner_to_coordinate"] = start_stop_frame_df["corner_path"].apply(
        lambda x: get_sleap_tracks_from_h5(x))
    start_stop_frame_df["corner_to_coordinate"] = start_stop_frame_df.apply(
        lambda x: {part: x["corner_to_coordinate"][:, index, :, :]
                   for index, part in enumerate(x["corner_parts"])},
        axis=1)
    start_stop_frame_df["corner_to_coordinate"] = start_stop_frame_df.apply(
        lambda x: {k: v[~np.isnan(v)][:2] for k, v in x["corner_to_coordinate"].items()}, axis=1)

    return start_stop_frame_df


def preprocess_start_stop_frame_data(start_stop_frame_df, sleap_dir):
    """
    Preprocesses the start/stop frame data.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the start/stop frame data.
        sleap_dir (str): The directory containing the SLEAP data.
    Returns:
        start_stop_frame_df (pandas dataframe): A dataframe containing the start/stop frame data preprocessed.
    """
    start_stop_frame_df = start_stop_frame_df.dropna(subset=["file_path"])
    start_stop_frame_df["sleap_name"] = start_stop_frame_df["file_path"].apply(
        lambda x: os.path.basename(x))
    start_stop_frame_df["video_name"] = start_stop_frame_df["file_path"].apply(
        lambda x: ".".join(os.path.basename(x).split(".")[:2]))
    start_stop_frame_df["start_frame"] = start_stop_frame_df["start_frame"].astype(
        int)
    start_stop_frame_df["stop_frame"] = start_stop_frame_df["stop_frame"].astype(
        int)
    start_stop_frame_df = start_stop_frame_df.drop(
        columns=["file_path", "notes"], errors="ignore")

    # Add any additional preprocessing steps here

    return start_stop_frame_df


def combine_with_lfp(start_stop_frame_df, lfp_spectral_df):
    """
    Combines the start/stop frame data with the LFP data.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the start/stop frame data.
        lfp_spectral_df (pandas dataframe): A dataframe containing the LFP data.
    Returns:
        lfp_and_sleap (pandas dataframe): A dataframe containing the start/stop frame data combined with the LFP data.
    """
    print("in combine func")
    start_stop_frame_df = start_stop_frame_df.dropna(subset=["video_name"])
    lfp_spectral_df = lfp_spectral_df.dropna(subset=["video_name"])
    lfp_and_sleap = pd.merge(
        start_stop_frame_df,
        lfp_spectral_df,
        on="video_name",
        how="inner")

    lfp_and_sleap["video_timestamps"].apply(lambda x: x.shape).head()
    lfp_and_sleap = lfp_and_sleap[lfp_and_sleap["subject_thorax_velocity"].apply(
        lambda x: np.isnan(x).any())]

    return lfp_and_sleap


def process_sleap_data(sleap_dir,
                       output_dir,
                       med_pc_width,
                       med_pc_height,
                       frame_rate,
                       window_size,
                       distance_threshold,
                       start_stop_frame_df,
                       lfp_spectral_df,
                       thorax_index,
                       output_prefix):
    """
    Processes the SLEAP data.
    Args:
        sleap_dir (str): The directory containing the SLEAP data.
        output_dir (str): The directory where the output data is saved.
        med_pc_width (float): The width of the video in pixels.
        med_pc_height (float): The height of the video in pixels.
        frame_rate (int): The frame rate of the video.
        window_size (int): The window size for the velocity calculation.
        distance_threshold (float): The distance threshold for the reward port.
        start_stop_frame_df (pandas dataframe): A dataframe containing the start/stop frame data.
        lfp_spectral_df (pandas dataframe): A dataframe containing the LFP data.
        thorax_index (int): The index of the thorax in the body parts list.
        output_prefix (str): Prefix for the output files.
    Returns:
        lfp_and_sleap (pandas dataframe): A dataframe containing the start/stop frame data combined with the LFP data.
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data.
    """
    # Set up paths and directories

    # Process start/stop frame data
    start_stop_frame_df = preprocess_start_stop_frame_data(
        start_stop_frame_df, sleap_dir)

    # Process sleap data
    start_stop_frame_df = process_sleap_tracks(
        start_stop_frame_df, sleap_dir, med_pc_width, med_pc_height)

    # Convert pixels to cm
    start_stop_frame_df = convert_pixels_to_cm(
        start_stop_frame_df, med_pc_width, med_pc_height)

    # Create individual columns for pose tracking
    start_stop_frame_df = create_individual_pose_tracking_columns(
        start_stop_frame_df)

    # Calculate velocity
    start_stop_frame_df = calculate_velocity(
        start_stop_frame_df, window_size, frame_rate, thorax_index)

    # Calculate distance to reward port
    start_stop_frame_df = calculate_distance_to_reward_port(
        start_stop_frame_df, thorax_index)

    # Combine with LFP data
    lfp_and_sleap = combine_with_lfp(start_stop_frame_df, lfp_spectral_df)
    print(lfp_and_sleap.columns)

    return lfp_and_sleap, start_stop_frame_df


def analyze_sleap_file(start_stop_frame_df, plot_output_dir,
                       output_prefix, thorax_index, thorax_plots, save_plots=False):
    """
    Analyzes the SLEAP files.
    Args:
        start_stop_frame_df (pandas dataframe): A dataframe containing the pose tracking data.
        plot_output_dir (str): Directory where the plots are saved.
        output_prefix (str): Prefix for the output files.
        thorax_index (int): The index of the thorax in the body parts list.
        thorax_plots (bool): Whether to plot the thorax data.
        save_plots (bool): Whether to save the plots.
    Returns:
        None
    """
    print("in describe_sleap_files")
    print(start_stop_frame_df.columns)
    print(start_stop_frame_df.head())
    for FILE_INDEX in range(len(start_stop_frame_df)):
        print(f"===File {FILE_INDEX}===")
        with h5py.File(start_stop_frame_df["sleap_path"].iloc[FILE_INDEX], "r") as f:
            dset_names = list(f.keys())
            current_subject = start_stop_frame_df["current_subject"].iloc[FILE_INDEX]
            locations = start_stop_frame_df["rescaled_locations"].iloc[FILE_INDEX][current_subject]
            node_names = [n.decode() for n in f["node_names"][:]]
        print("===HDF5 datasets===")
        print(dset_names)
        print()

        print("===locations data shape===")
        print(locations.shape)
        print()

        print("===nodes===")
        for i, name in enumerate(node_names):
            print(f"{i}: {name}")
        print()

    if thorax_plots:
        # TODO: thorax index is hard coded

        # Thorax location
        print("getting thorax loc")
        thorax_loc = locations[:, thorax_index, :]
        print(thorax_loc)
        fig, ax = plt.subplots()

        plt.plot(thorax_loc[:, 0], label='X-coordinates')
        # Converting to negative so that we can see both x and y track
        plt.plot(-1 * thorax_loc[:, 1], label='Y-coordinates')

        plt.legend(loc="center right")
        plt.title('Thorax locations')
        plt.xlabel("Time in frames")
        plt.ylabel("Coordinate Position")
        plt.show()

        if save_plots:
            print("in save blot")
            plt.savefig(
                os.path.join(
                    plot_output_dir,
                    f"{output_prefix}_thorax_locations.png"))

        # Thorax tracks
        plt.figure(figsize=(7, 7))
        plt.plot(thorax_loc[:, 0], thorax_loc[:, 1])
        print("line 1971")
        plt.title('Thorax tracks')
        plt.xlabel("X-Coordinates")
        plt.ylabel("Y-Coordinates")

        if save_plots:
            plt.savefig(
                os.path.join(
                    plot_output_dir,
                    f"{output_prefix}_thorax_tracks.png"))


def read_phy_data(all_phy_dir, sampling_rate, lfp_spectral):
    """
    Read and process data from Phy.
    """
    recording_to_cluster_info = {}
    print("in read_phy_data")
    print(all_phy_dir)

    for recording_dir in all_phy_dir:
        try:
            recording_basename = os.path.basename(recording_dir).strip(".rec")
            print(recording_basename)

            # Read cluster info
            file_path = os.path.join(recording_dir, "phy", "cluster_info.tsv")
            recording_to_cluster_info[recording_basename] = pd.read_csv(
                file_path, sep="\t")
        except Exception as e:
            print(e)

    recording_to_cluster_info_df = pd.concat(
        recording_to_cluster_info, names=['recording_name']).reset_index(level=1, drop=True).reset_index()

    good_unit_cluster_info_df = recording_to_cluster_info_df[
        recording_to_cluster_info_df["group"] == "good"].reset_index(drop=True)

    recording_to_good_unit_ids = good_unit_cluster_info_df.groupby(
        'recording_name')['cluster_id'].apply(list).to_dict()

    recording_to_spike_clusters = {}

    for recording_dir in all_phy_dir:
        try:
            recording_basename = os.path.basename(recording_dir).strip(".rec")
            file_path = os.path.join(
                recording_dir, "phy", "spike_clusters.npy")
            recording_to_spike_clusters[recording_basename] = np.load(
                file_path)
        except Exception as e:
            print(e)

    recording_to_spike_times = {}
    for recording_dir in all_phy_dir:
        try:
            recording_basename = os.path.basename(recording_dir).strip(".rec")
            file_path = os.path.join(recording_dir, "phy", "spike_times.npy")
            recording_to_spike_times[recording_basename] = np.load(file_path)
        except Exception as e:
            print(e)

    recording_to_spike_df = {}

    for recording_dir in all_phy_dir:
        print("in line 2033")
        try:
            print("in try")
            recording_basename = os.path.basename(recording_dir).strip(".rec")
            cluster_info_path = os.path.join(
                recording_dir, "phy", "cluster_info.tsv")
            cluster_info_df = pd.read_csv(cluster_info_path, sep="\t")

            spike_clusters_path = os.path.join(
                recording_dir, "phy", "spike_clusters.npy")
            spike_clusters = np.load(spike_clusters_path)

            spike_times_path = os.path.join(
                recording_dir, "phy", "spike_times.npy")
            spike_times = np.load(spike_times_path)

            curr_spike_df = pd.DataFrame(
                {'spike_clusters': spike_clusters, 'spike_times': spike_times.T[0]})

            merged_df = curr_spike_df.merge(cluster_info_df, left_on='spike_clusters', right_on='cluster_id',
                                            how="left")
            merged_df["recording_name"] = recording_basename
            print(merged_df.columns)
            print(merged_df.groupby('spike_clusters')["spike_times"].diff())
            print(type(merged_df.groupby(
                'spike_clusters')["spike_times"].diff()))
            print(merged_df.groupby('spike_clusters')
                  ["spike_times"].diff().index)
            print(type(merged_df.groupby('spike_clusters')
                  ["spike_times"].diff().values))
            # set timestamp_isi to the list of diff
            merged_df["timestamp_isi"] = merged_df.groupby(
                'spike_clusters')["spike_times"].diff().values
            merged_df["timestamp_isi"] = merged_df["timestamp_isi"].fillna(0)
            merged_df["timestamp_isi"] = merged_df["timestamp_isi"].astype(
                float)
            print("sampling rate is: ", sampling_rate)
            merged_df["timestamp_isi"] = merged_df["timestamp_isi"].apply(
                lambda x: float(x) / float(sampling_rate))
            # divide by sampling rate to get the current isi using lambda function

            merged_df["current_isi"] = merged_df["timestamp_isi"]

            if not merged_df.empty:
                recording_to_spike_df[recording_basename] = merged_df

        except Exception as e:
            print(e)
        print(recording_to_spike_df)
        all_spike_time_df = pd.concat(recording_to_spike_df.values())
        all_spike_time_df = all_spike_time_df[all_spike_time_df["group"] == "good"].reset_index(
            drop=True)

        grouped_df = all_spike_time_df.groupby(['spike_clusters', 'recording_name'])["spike_times"].apply(
            lambda x: np.array(x)).reset_index()
        grouped_df = grouped_df.sort_values(
            by=['recording_name', 'spike_clusters']).reset_index(drop=True)
        max_number_of_spikes = all_spike_time_df["n_spikes"].max()
        grouped_df["spike_times"] = grouped_df["spike_times"].apply(
            lambda x: np.concatenate([x, np.full([max_number_of_spikes - x.shape[0]], np.nan)]))
        grouped_df = grouped_df.groupby('recording_name').agg(
            {'spike_clusters': lambda x: list(x), 'spike_times': lambda x: np.array(list(x))}).reset_index()

        lfp_spectral_df = pd.merge(lfp_spectral, grouped_df, left_on='recording', right_on="recording_name",
                                   how='inner')

    return lfp_spectral_df, grouped_df, all_spike_time_df


def filter_good_units(recording_to_cluster_info):
    """
    Filter good units from cluster information.
    """
    print(recording_to_cluster_info)
    recording_to_cluster_info_df = pd.concat(recording_to_cluster_info, names=['recording_name']).reset_index(level=1,
                                                                                                              drop=True).reset_index()
    good_unit_cluster_info_df = recording_to_cluster_info_df[
        recording_to_cluster_info_df["group"] == "good"].reset_index(drop=True)
    recording_to_good_unit_ids = good_unit_cluster_info_df.groupby(
        'recording_name')['cluster_id'].apply(list).to_dict()

    return recording_to_good_unit_ids


def calculate_rolling_avg_firing_rate(firing_times, window_size=2000, slide=2000, stop_time=None):
    """
    Calculates the rolling average firing rate of a neuron.

    Parameters:
    - firing_times (numpy.ndarray): An array of firing times of a neuron.
    - window_size (int or float): The size of the window for calculating the average firing rate.
    - slide (int or float): The amount to slide the window for each calculation.
    - stop_time (int or float, optional): The timestamp to stop the calculation at. If None, the calculation goes until the end of the firing times.

    Returns:
    - tuple: Two numpy.ndarrays, the rolling average firing rates and the starting timestamps of each window.
    """
    # Convert firing_times to a numpy array if it's not already
    firing_times = np.array(firing_times)

    # Initialize lists to store the results
    avg_firing_rates = []
    window_starts = []

    # If no stop_time is provided, use the last firing time
    if stop_time is None:
        stop_time = firing_times[-1]

    # Calculate the number of windows
    num_windows = int((stop_time - window_size) / slide) + 1

    # Loop over each window
    for i in range(num_windows):
        # Calculate the start and end of the window
        start = i * slide
        end = start + window_size

        # Calculate the average firing rate for this window
        firing_rate = np.sum((firing_times >= start) &
                             (firing_times < end)) / window_size

        # Store the results
        avg_firing_rates.append(firing_rate)
        window_starts.append(start)

    return np.array(avg_firing_rates), np.array(window_starts)


def calculate_firing_rates(lfp_spectral_df, grouped_df,
                           sampling_rate, spike_window):
    """
    Calculate firing rates and timestamps for neurons.
    """
    print("PRINTING COLS")
    print("LFP SPECTRAL DF")
    for col in lfp_spectral_df.columns:
        print(col)
    print("\n\n")
    print("GROUPED DF")
    for col in grouped_df.columns:
        print(col)
    """
    lfp_spectral_df = pd.merge(
        lfp_spectral_df,
        grouped_df,
        left_on='recording',
        right_on="recording_name",
        how='inner')
        """
    print("COLUMNS IN MERGED")
    # check if last_timestamp is in the columns
    last_timestamp = False
    for col in lfp_spectral_df.columns:
        if "last_timestamp" in col:
            last_timestamp = True
    if not last_timestamp:
        lfp_spectral_df["last_timestamp"] = lfp_spectral_df["raw_timestamps"].apply(
            lambda x: x[-1])

    lfp_spectral_df["neuron_average_fr"] = lfp_spectral_df.apply(
        lambda x: np.array([calculate_rolling_avg_firing_rate(np.array(times[~np.isnan(times)]),
                                                              stop_time=x["last_timestamp"] -
                                                              x["first_timestamp"],
                                                              window_size=spike_window,
                                                              slide=spike_window)[0] for times in x["spike_times"][0]]), axis=1)

    lfp_spectral_df["neuron_average_timestamps"] = lfp_spectral_df.apply(lambda x: calculate_rolling_avg_firing_rate(
        x["spike_times"][0][
            ~np.isnan(x["spike_times"][0])],
        stop_time=x["last_timestamp"] - x[
            "first_timestamp"],
        window_size=spike_window,
        slide=spike_window)[1], axis=1)

    lfp_spectral_df["neuron_average_fr"] = lfp_spectral_df.apply(lambda x: x["neuron_average_fr"] * spike_window,
                                                                 axis=1)

    return lfp_spectral_df


def add_spike_to_phy(phy_curation_path, lfp_spectral_df, sampling_rate,
                     output_dir, output_prefix):
    """
    Adds the spike data to the Phy curation file.
    Args:
        phy_curation_path (list: str): The path to the Phy curation file.
        lfp_spectral_df (pandas dataframe): A dataframe containing the LFP data.
        output_dir (str): Directory where the output data is saved.
        output_prefix (str): Prefix for the output files.
    Returns:
        lfp_spectral_df (pandas dataframe): A dataframe containing the LFP data with the spike data added.
        grouped_df (pandas dataframe): A dataframe containing the grouped neurons.
        all_spike_time_df (pandas dataframe): A dataframe containing all the spike times.
    """
    # Read Phy curation file with read function

    lfp_spectral_df, grouped_df, all_spike_time_df = read_phy_data(all_phy_dir=phy_curation_path,
                                                                   lfp_spectral=lfp_spectral_df,
                                                                   sampling_rate=sampling_rate)

    lfp_spectral_df = calculate_firing_rates(lfp_spectral_df=lfp_spectral_df, grouped_df=grouped_df,
                                             sampling_rate=sampling_rate, spike_window=1)

    # save the results
    print("SAVING  IN ADD")
    lfp_spectral_df.to_pickle(output_dir + "/lfpspectral_df.pkl")
    grouped_df.to_pickle(output_dir + "/grouped_df.pkl")
    all_spike_time_df.to_pickle(output_dir + "/all_spike_time_df.pkl")

    return lfp_spectral_df, grouped_df, all_spike_time_df


def make_labels_df(labels_df, filter_bands_df):
    # TODO: this function can be optimized, for loops

    filter_bands_df["video_name"] = filter_bands_df["video_name"].apply(
        lambda x: x.strip(".videoTimeStamps.cameraHWSync"))
    filter_bands_df["current_subject"] = filter_bands_df["current_subject"].astype(
        str)
    labels_df["current_subject"] = labels_df["current_subject"].astype(str)

    # merge
    trial_and_spectral = pd.merge(labels_df, filter_bands_df, on=["current_subject", "video_name"],
                                  how="inner", suffixes=('', '_y'))
    # todo: hardcoded values, double check
    trial_and_spectral["baseline_start_timestamp"] = trial_and_spectral["tone_start_timestamp"] - 30 * 20000
    trial_and_spectral["baseline_stop_timestamp"] = trial_and_spectral["tone_start_timestamp"] - 20 * 20000

    power_columns = [col for col in trial_and_spectral.columns if
                     "power" in col and "timestamps" not in col and "calculation" not in col]

    for col in power_columns:
        brain_region = col.replace("all_windows", "_").strip("_")
        print(brain_region)

        updated_item_col = "baseline_{}".format(brain_region)
        updated_timestamp_col = "baseline_power_timestamps".format(
            brain_region)
        trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                        helper_filter_by_timestamp_range(
                                                                            start=x[
                                                                                "baseline_start_timestamp"],
                                                                            stop=x[
                                                                                "baseline_stop_timestamp"],
                                                                            timestamps=x[
                                                                                "power_timestamps"],
                                                                            items=x[col])[1], axis=1)
    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x[
                                                                                 "baseline_start_timestamp"],
                                                                             stop=x[
                                                                                 "baseline_stop_timestamp"],
                                                                             timestamps=x[
                                                                                 "power_timestamps"],
                                                                             items=x[col])[0], axis=1)

    for col in power_columns:
        brain_region = col.replace("all_windows", "_").strip("_")
        print(brain_region)

        updated_item_col = "trial_{}".format(brain_region)
        updated_timestamp_col = "trial_power_timestamps".format(brain_region)
        trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                        helper_filter_by_timestamp_range(
                                                                            start=x["tone_start_timestamp"],
                                                                            stop=x["tone_stop_timestamp"],
                                                                            timestamps=x["power_timestamps"],
                                                                            items=x[col])[1], axis=1)
    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x["tone_start_timestamp"],
                                                                             stop=x["tone_stop_timestamp"],
                                                                             timestamps=x["power_timestamps"],
                                                                             items=x[col])[0], axis=1)

    coherence_columns = [col for col in trial_and_spectral.columns if
                         "coherence" in col and "timestamps" not in col and "calculation" not in col]
    for col in coherence_columns:
        brain_region = col.replace("all_windows", "_").strip("_")
        print(brain_region)

        updated_item_col = "baseline_{}".format(brain_region)
        updated_timestamp_col = "baseline_coherence_timestamps".format(
            brain_region)
        trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                        helper_filter_by_timestamp_range(
                                                                            start=x["baseline_start_timestamp"],
                                                                            stop=x["baseline_stop_timestamp"],
                                                                            timestamps=x[
                                                                                "coherence_timestamps"],
                                                                            items=x[col])[1], axis=1)
    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x[
                                                                                 "baseline_start_timestamp"],
                                                                             stop=x["baseline_stop_timestamp"],
                                                                             timestamps=x[
                                                                                 "coherence_timestamps"],
                                                                             items=x[col])[0], axis=1)
    for col in coherence_columns:
        brain_region = col.replace("all_windows", "_").strip("_")
        print(brain_region)

        updated_item_col = "trial_{}".format(brain_region)
        updated_timestamp_col = "trial_coherence_timestamps".format(
            brain_region)
        trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                        helper_filter_by_timestamp_range(
                                                                            start=x["tone_start_timestamp"],
                                                                            stop=x["tone_stop_timestamp"],
                                                                            timestamps=x[
                                                                                "coherence_timestamps"],
                                                                            items=x[col])[1], axis=1)
    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x["tone_start_timestamp"],
                                                                             stop=x["tone_stop_timestamp"],
                                                                             timestamps=x[
                                                                                 "coherence_timestamps"],
                                                                             items=x[col])[0], axis=1)

    granger_columns = [col for col in trial_and_spectral.columns if
                       "granger" in col and "timestamps" not in col and "calculation" not in col]

    for col in granger_columns:
        brain_region = col.replace("all_windows", "_").strip("_")
        print(brain_region)

        updated_item_col = "baseline_{}".format(brain_region)
        updated_timestamp_col = "baseline_granger_timestamps".format(
            brain_region)
        trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                        helper_filter_by_timestamp_range(
                                                                            start=x["baseline_start_timestamp"],
                                                                            stop=x["baseline_stop_timestamp"],
                                                                            timestamps=x["granger_timestamps"],
                                                                            items=x[col])[1], axis=1)
    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x[
                                                                                 "baseline_start_timestamp"],
                                                                             stop=x["baseline_stop_timestamp"],
                                                                             timestamps=x["granger_timestamps"],
                                                                             items=x[col])[0], axis=1)
    for col in granger_columns:
        brain_region = col.replace("all_windows", "_").strip("_")
        print(brain_region)

        updated_item_col = "trial_{}".format(brain_region)
        updated_timestamp_col = "trial_granger_timestamps".format(brain_region)
        trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                        helper_filter_by_timestamp_range(
                                                                            start=x["tone_start_timestamp"],
                                                                            stop=x["tone_stop_timestamp"],
                                                                            timestamps=x["granger_timestamps"],
                                                                            items=x[col])[1], axis=1)
    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x["tone_start_timestamp"],
                                                                             stop=x["tone_stop_timestamp"],
                                                                             timestamps=x["granger_timestamps"],
                                                                             items=x[col])[0], axis=1)

    sleap_columns = [col for col in trial_and_spectral.columns if
                     "locations" in col or "velocity" in col or "to_reward_port" in col]

    for col in sorted(sleap_columns):
        updated_item_col = "baseline_{}".format(col)
        print(updated_item_col)
        updated_timestamp_col = "baseline_video_timestamps".format(col)
        if "agent" in col:
            trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                            helper_filter_by_timestamp_range(
                                                                                start=x[
                                                                                    "baseline_start_timestamp"],
                                                                                stop=x[
                                                                                    "baseline_stop_timestamp"],
                                                                                timestamps=x[
                                                                                    "video_timestamps"],
                                                                                items=x[col])[1] if x[
                                                                                "agent"] else np.nan, axis=1)
        else:
            trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                            helper_filter_by_timestamp_range(
                                                                                start=x[
                                                                                    "baseline_start_timestamp"],
                                                                                stop=x[
                                                                                    "baseline_stop_timestamp"],
                                                                                timestamps=x[
                                                                                    "video_timestamps"],
                                                                                items=x[col])[1], axis=1)

    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x[
                                                                                 "baseline_start_timestamp"],
                                                                             stop=x["baseline_stop_timestamp"],
                                                                             timestamps=x["video_timestamps"],
                                                                             items=x[col])[0], axis=1)

    for col in sorted(sleap_columns):
        updated_item_col = "trial_{}".format(col)
        print(updated_item_col)
        updated_timestamp_col = "trial_video_timestamps".format(col)
        if "agent" in col:
            trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                            helper_filter_by_timestamp_range(
                                                                                start=x["tone_start_timestamp"],
                                                                                stop=x["tone_stop_timestamp"],
                                                                                timestamps=x[
                                                                                    "video_timestamps"],
                                                                                items=x[col])[1] if x[
                                                                                "agent"] else np.nan, axis=1)
        else:
            trial_and_spectral[updated_item_col] = trial_and_spectral.apply(lambda x:
                                                                            helper_filter_by_timestamp_range(
                                                                                start=x["tone_start_timestamp"],
                                                                                stop=x["tone_stop_timestamp"],
                                                                                timestamps=x[
                                                                                    "video_timestamps"],
                                                                                items=x[col])[1], axis=1)

    trial_and_spectral[updated_timestamp_col] = trial_and_spectral.apply(lambda x:
                                                                         helper_filter_by_timestamp_range(
                                                                             start=x["tone_start_timestamp"],
                                                                             stop=x["tone_stop_timestamp"],
                                                                             timestamps=x["video_timestamps"],
                                                                             items=x[col])[0], axis=1)

    return trial_and_spectral


def encode_labels(filter_bands_df, labels_df, encoding_dict):
    # TODO: check, this function (based on 11_rce2_export) is identical to make_labels_df (based on 10_rce2_add_trial_labels)
    filter_bands_df["competition_closeness"] = filter_bands_df["competition_closeness"].map(
        encoding_dict)
    filter_bands_df["tracked_subject"] = filter_bands_df["tracked_subject"].apply(
        lambda x: "_".join([str(subj) for subj in x]))
    filter_bands_df["current_tracked_subject"] = filter_bands_df["tracked_subject"]

    # Example with multiple columns being aggregated
    group_labels_df = labels_df.groupby(['video_name', "current_subject"]).agg(
        {col: list for col in [_ for _ in labels_df.columns if _ not in ["video_name", 'current_subject']]
         # Aggregating into a list
         # Other columns can be aggregated differently here
         }).reset_index()
    group_labels_df["session_dir"] = group_labels_df["session_dir"].apply(
        lambda x: x[0])
    group_labels_df["sleap_name"] = group_labels_df["sleap_name"].apply(
        lambda x: x[0])
    group_labels_df["trial_notes"] = group_labels_df["notes"]

    trial_and_spectral_df = pd.merge(group_labels_df, filter_bands_df, on=["current_subject", "video_name"],
                                     how="inner", suffixes=('', '_y'))

    trial_and_spectral_df = trial_and_spectral_df.drop(
        columns=[col for col in trial_and_spectral_df.columns if "_y" in col], errors="ignore")

    trial_and_spectral_df["start_stop_frame"] = trial_and_spectral_df.apply(
        lambda x: (x["start_frame"] - 1, x["stop_frame"] - 1), axis=1)

    trial_and_spectral_df["start_stop_timestamps"] = trial_and_spectral_df.apply(
        lambda x: helper_extract_start_stop_elements(x["video_timestamps"], x["start_stop_frame"][0],
                                                     x["start_stop_frame"][1]), axis=1)

    sleap_columns = ["subject_locations",
                     "agent_locations",
                     "subject_thorax_velocity",
                     "agent_thorax_velocity",
                     "subject_thorax_to_reward_port",
                     "agent_thorax_to_reward_port"]

    for col in sleap_columns:
        trial_and_spectral_df[col] = trial_and_spectral_df.apply(lambda x: helper_mask_slices(x[col], x["start_stop_frame"]),
                                                                 axis=1)

    trial_and_spectral_df = trial_and_spectral_df.drop(columns=["tone_frames", "start_frame", "stop_frame"],
                                                       errors="ignore")

    trial_and_spectral_df["tone_mask"] = trial_and_spectral_df.apply(
        lambda x: helper_filter_array_by_values(x["tone_start_frame"],
                                                x["start_stop_frame"][0],
                                                x["start_stop_frame"][1])[1], axis=1)

    for column in [col for col in trial_and_spectral_df if "frame" in col and col != "start_stop_frame"]:
        trial_and_spectral_df[column] = trial_and_spectral_df.apply(lambda x: helper_filter_array_by_values(x[column],
                                                                                                            x[
            "start_stop_frame"][
            0],
            x[
            "start_stop_frame"][
            1])[0],
            axis=1)
    trial_and_spectral_df["condition"] = trial_and_spectral_df.apply(
        lambda x: np.array(x["condition"])[x["tone_mask"]], axis=1)

    trial_and_spectral_df["tracked_subject"] = trial_and_spectral_df["tracked_subject"].apply(
        lambda x: x[0])
    trial_and_spectral_df["box_number"] = trial_and_spectral_df["box_number"].apply(
        lambda x: x[0])
    trial_and_spectral_df["experiment"] = trial_and_spectral_df["experiment"].apply(
        lambda x: x[0])

    for col in sorted(sleap_columns):
        updated_item_col = "baseline_{}".format(col)
        print(updated_item_col)
        updated_timestamp_col = "baseline_video_timestamps".format(col)
        if "agent" in col:
            trial_and_spectral_df[updated_item_col] = trial_and_spectral_df.apply(lambda x:
                                                                                  helper_filter_by_timestamp_range(
                                                                                      start=x[
                                                                                          "baseline_start_timestamp"],
                                                                                      stop=x[
                                                                                          "baseline_stop_timestamp"],
                                                                                      timestamps=x[
                                                                                          "video_timestamps"],
                                                                                      items=x[col])[1] if x[
                                                                                      "agent"] else np.nan, axis=1)
        else:
            trial_and_spectral_df[updated_item_col] = trial_and_spectral_df.apply(lambda x:
                                                                                  helper_filter_by_timestamp_range(
                                                                                      start=x[
                                                                                          "baseline_start_timestamp"],
                                                                                      stop=x[
                                                                                          "baseline_stop_timestamp"],
                                                                                      timestamps=x[
                                                                                          "video_timestamps"],
                                                                                      items=x[col])[1], axis=1)

    trial_and_spectral_df[updated_timestamp_col] = trial_and_spectral_df.apply(lambda x:
                                                                               helper_filter_by_timestamp_range(
                                                                                   start=x[
                                                                                       "baseline_start_timestamp"],
                                                                                   stop=x["baseline_stop_timestamp"],
                                                                                   timestamps=x["video_timestamps"],
                                                                                   items=x[col])[0], axis=1)

    aggregation_dict = {col: "first" for col in trial_and_spectral_df if
                        col not in ['subject_locations', "current_subject", "session_dir"]}
    aggregation_dict["subject_locations"] = list
    trial_and_spectral_df = trial_and_spectral_df.groupby(["current_subject", "session_dir"]).agg(
        aggregation_dict).reset_index()

    helper_combine_grouped_rows(trial_and_spectral_df, ["subject_locations"])

    helper_overlay_arrays(trial_and_spectral_df["subject_locations"].iloc[0],
                          trial_and_spectral_df["subject_locations"].iloc[1])

    trial_and_spectral_df["baseline_start_timestamp"] = trial_and_spectral_df["tone_start_timestamp"] - 30 * 20000
    trial_and_spectral_df["baseline_stop_timestamp"] = trial_and_spectral_df["tone_start_timestamp"] - 20 * 20000

    trial_and_spectral_df["video_frame"] = trial_and_spectral_df["video_timestamps"].apply(
        lambda x: np.array(list(range(len(x)))) + 1)

    return trial_and_spectral_df
