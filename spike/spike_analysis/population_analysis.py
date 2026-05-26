import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, pdist
from scipy.spatial import procrustes as scipy_procrustes
from scipy.linalg import orthogonal_procrustes
from scipy.stats import spearmanr
from itertools import combinations
import spike.spike_analysis.spike_collection as col
import spike.spike_analysis.spike_recording
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.multitest import multipletests


def get_indices(repeated_items_list):
    """
    Takes in a list of repeated items, creates a list of indices that correspond to each unique item chunk.

    Args (1):
        repeated_items_list: list, list of repeated items

    Returns:
        result: list of tuples, where the first element
            is the first index of a unique item, and the second
            element is the last index of that unique item
    """
    result = []
    start = 0
    current = repeated_items_list[0]
    for i, item in enumerate(repeated_items_list[1:], 1):
        if item != current:
            result.append([start, i - 1])
            start = i
            current = item
    # Don't forget the last group
    result.append([start, len(repeated_items_list) - 1])
    return result


def event_slice(transformed_subsets, key, no_PCs):
    """
    Takes in a matrix of PCA embedded firing rates for multiple events
    and an event key (event labels per timebin) and the number of PC's to use
    to calculate the geodesic distance with across event types.

    Args (3):
        transformed_subsets: np.array, d[session X timebin X PCS] or [timebins x pcs]
        key: list of str, each element is an event type and
            corresponds to the timebin dimension indices of
            the transformed_subsets matrix
        no_PCs: int, number of PCs required to explain a variance threshold
        mode: {'multisession', 'single'}; multisession calculates event slices
            for multiple sessions worth of firing rates, single calculates event slices for a
            single sessions worth of firing rates
    Returns:
        trajectories: dict, events to trajectories across each PCA embedding
            keys: str, event types
            values: np.array, d=[session x timebins x no_PCs] or [timebins x PCs]
    """
    event_indices = get_indices(key)
    events = np.unique(key)
    trajectories = {}
    for i in range(len(event_indices)):
        event = events[i]
        start = event_indices[i][0]
        stop = event_indices[i][1]
        if len(transformed_subsets.shape) == 3:
            event_trajectory = transformed_subsets[:, start : stop + 1, :no_PCs]
        if len(transformed_subsets.shape) == 2:
            event_trajectory = transformed_subsets[start : stop + 1, :no_PCs]
        trajectories[event] = event_trajectory
    return trajectories


def euclidean_distances(event_trajectories, recording_name=None, evr=None):
    """
    Calculates the euclidean distances between all trajectories in the event_trajectory dictionary,

    Arguments(1 required, 3 total):
        event_trajectories: dictionary
            keys: str, event names
            values: numpy arrays of shape [session x timebins x PCs] or [timebins x PCs]
        recording_name: str, optional index labeled for the resulting dataframe
        evr: np.array, optional explained variance ratios for each PC used as weights.
            If provided, distances are weighted: d = sqrt(Σ evr_i * (x_i - y_i)²).
            Should be sliced to match the number of PCs in event_trajectories.

    Returns (1):
        df: DataFrame, columns are event pairs and data is a list of disntaces, or a single distance between trajectories
    """
    # Get all event pairs
    event_pairs = list(combinations(event_trajectories.keys(), 2))

    # Calculate distances for each pair
    distances = []
    for pair in event_pairs:
        event1 = event_trajectories[pair[0]]
        event2 = event_trajectories[pair[1]]
        dist = distance_bw_trajectories(event1, event2, evr=evr)
        distances.append(dist)

    # Create column names from pairs
    column_names = [f"{pair[0]}_{pair[1]}" for pair in event_pairs]
    # Create DataFrame
    if recording_name is not None:
        df = pd.DataFrame([distances], columns=column_names, index=[recording_name])

    else:
        df = pd.DataFrame([distances], columns=column_names)

    return df


def distance_bw_trajectories(trajectory1, trajectory2, evr=None):
    """
    Calculates the geodesic distance between two event trajectories by summing the distance between
    congruent timebins across trajectories.

    Arugments (2 required, 3 total):
        trajectory1 & trajectory2: numpy ararys of shape [session x timebin x PCs] pr [timebin x PCs]
        evr: np.array, optional explained variance ratios used as PC weights.
            If provided, each timebin distance is weighted: d = sqrt(Σ evr_i * (x_i - y_i)²).
            Should be sliced to match the number of PCs in trajectory1/trajectory2.

    Returns (1):
        euclidean_distances: either a single value for 1 session's trajectories, or a list of distances across
        all sessions trajectories
    """
    if len(trajectory1.shape) == 3:
        euclidean_distances = []
        for session in range(trajectory1.shape[0]):
            dist_bw_tb = 0
            for i in range(trajectory1.shape[1]):
                dist_bw_tb = dist_bw_tb + euclidean(trajectory1[session, i, :], trajectory2[session, i, :], w=evr)
            euclidean_distances.append(dist_bw_tb)
    if len(trajectory1.shape) == 2:
        dist_bw_tb = 0
        for i in range(trajectory1.shape[0]):
            dist_bw_tb = dist_bw_tb + euclidean(trajectory1[i, :], trajectory2[i, :], w=evr)
        euclidean_distances = dist_bw_tb
    return euclidean_distances


def PCs_needed(explained_variance_ratios, percent_explained=0.9):
    """
    Calculates number of principle compoenents needed given a percent
    variance explained threshold.

    Args(2 total, 1 required):
        explained_variance_ratios: np.array,
            output of pca.explained_variance_ratio_
        percent_explained: float, default=0.9, percent
        variance explained threshold

    Return:
        i: int, number of principle components needed to
           explain percent_explained variance
    """
    for i in range(len(explained_variance_ratios)):
        if explained_variance_ratios[0:i].sum() > percent_explained:
            return i


def avg_traj(event_firing_rates, num_points, events):
    event_averages = np.nanmean(event_firing_rates, axis=0)
    event_keys = [event for event in events for _ in range(num_points)]
    return event_averages, event_keys


def trial_traj(event_firing_rates, num_points, min_event):
    trials, timebins, units = event_firing_rates.shape
    num_data_ps = num_points * min_event
    event_firing_rates = event_firing_rates[:min_event, :, :]
    event_firing_rates_conc = event_firing_rates.reshape(min_event * timebins, units)
    return event_firing_rates_conc, num_data_ps


def check_recording(recording, min_neurons, events, to_print=True):
    if recording.analyzed_neurons < min_neurons:
        if to_print:
            print(f"Excluding {recording.name} with {recording.good_neurons} neurons")
        return False
    for event in events:
        events_array = recording.event_dict[event]

        # check 1: nothing in the array at all
        if events_array is None or events_array.size == 0:
            if to_print:
                print(f"Excluding {recording.name}, it has no {event} events")
            return False

        # check 2: only one event and it has zero duration
        if len(events_array) == 1:
            start, end = events_array[0][0], events_array[0][1]
            if end - start == 0:
                if to_print:
                    print(f"Excluding {recording.name}, it has a zero-length {event} event")
                return False
    return True


def pca_matrix(
    spike_collection,
    event_length,
    pre_window,
    post_window,
    events,
    mode,
    min_neurons=0,
    min_events=None,
    condition_dict=None,
):
    event_keys = []
    recording_keys = []
    pca_master_matrix = None
    event_count = {}
    if isinstance(spike_collection, col.SpikeCollection):
        recordings = spike_collection.recordings
        timebin = spike_collection.timebin
        if events is None:
            events = spike_collection.recordings[0].event_dict.keys()
    elif isinstance(spike_collection, list):
        recordings = spike_collection
        timebin = spike_collection[0].timebin
        if events is None:
            events = spike_collection[0].event_dict.keys()
    else:
        recordings = [spike_collection]
        timebin = spike_collection.timebin
        if events is None:
            events = spike_collection.event_dict.keys()

    num_points = int((event_length + pre_window + post_window) * 1000 / timebin)
    for recording in recordings:
        recording_good = check_recording(recording, min_neurons, events, to_print=True)
        if recording_good:
            event_count[recording.name] = {}
            pca_matrix = None
            for event in events:
                firing_rates = recording.event_firing_rates(event, event_length, pre_window, post_window)
                event_count[recording.name][event] = len(firing_rates)
                if mode == "average":
                    event_firing_rates, event_keys = avg_traj(firing_rates, num_points, events)
                if mode == "trial":
                    min_event = min_events[event]
                    event_firing_rates, num_data_ps = trial_traj(firing_rates, num_points, min_event)
                    if pca_master_matrix is None:
                        event_keys.extend([event] * num_data_ps)
                if pca_matrix is not None:
                    # event_firing_rates = timebins, neurons
                    pca_matrix = np.concatenate((pca_matrix, event_firing_rates), axis=0)
                if pca_matrix is None:
                    pca_matrix = event_firing_rates
            if pca_master_matrix is not None:
                pca_master_matrix = np.concatenate((pca_master_matrix, pca_matrix), axis=1)
            if pca_master_matrix is None:
                pca_master_matrix = pca_matrix
            recording_keys.extend([recording.name] * pca_matrix.shape[1])
        # timebins by neurons
    if pca_master_matrix is not None:
        return PCAResult(
            spike_collection=spike_collection,
            event_length=event_length,
            pre_window=pre_window,
            post_window=post_window,
            raw_data=pca_master_matrix,
            recording_keys=recording_keys,
            event_keys=event_keys,
            event_count=event_count,
            condition_dict=condition_dict,
        )
    else:
        return None


def avg_trajectory_matrix(
    spike_collection, event_length, pre_window, post_window=0, events=None, min_neurons=0, condition_dict=None
):
    """
    Args (5 total, 2 required):
        event_length: int, length (s) of event transformed by PCA
        pre_window: int, length (s) of time prior to event onset included in PCA
        post_window: int, default=0, length(s) of time after event_length (s) included in PCA
        events: list of str, default=None, event types for PCA to be applied on their firing
            rate averages, if no list given, PCA is applied on all event types in event_dict

    Returns:
        PCA_dict

    """
    return pca_matrix(
        spike_collection,
        event_length,
        pre_window,
        post_window,
        events,
        mode="average",
        min_neurons=min_neurons,
        min_events=None,
        condition_dict=condition_dict,
    )


def trial_trajectory_matrix(spike_collection, event_length, pre_window, post_window=0, events=None, min_neurons=0):
    """
    Args (5 total, 2 required):
        event_length: int, length (s) of event transformed by PCA
        pre_window: int, length (s) of time prior to event onset included in PCA
        post_window: int, default=0, length(s) of time after event_length (s) included in PCA
        events: list of str, default=None, event types for PCA to be applied on their firing
            rate averages, if no list given, PCA is applied on all event types in event_dict

    Returns:
        PCA_dict

    """
    min_events = event_numbers(spike_collection, events, min_neurons)
    return pca_matrix(
        spike_collection,
        event_length,
        pre_window,
        post_window,
        events,
        mode="trial",
        min_neurons=min_neurons,
        min_events=min_events,
    )


def event_numbers(spike_collection, events, min_neurons, to_print=False):
    mins = {}
    if events is None:
        events = list(spike_collection.recordings[0].event_dict.keys())
    for event in events:
        totals = []
        for recording in spike_collection.recordings:
            recording_good = check_recording(recording, min_neurons, events, to_print=False)
            if recording_good:
                totals.append((recording.event_dict[event]).shape[0])
        mins[event] = min(totals)
    return mins


class PCAResult:
    def __init__(
        self,
        spike_collection,
        event_length,
        pre_window,
        post_window,
        raw_data,
        recording_keys,
        event_keys,
        event_count,
        condition_dict,
    ):

        self.raw_data = raw_data
        matrix_df = pd.DataFrame(data=raw_data, columns=recording_keys, index=event_keys)
        self.matrix_df = matrix_df
        try:
            self.timebin = spike_collection.timebin
        except AttributeError:
            self.timebin = spike_collection[0].timebin
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.recordings = list(matrix_df.columns.unique())
        self.events = list(matrix_df.index.unique())
        self.labels = np.array(matrix_df.index.to_list())
        if raw_data.shape[0] < raw_data.shape[1]:
            print("Warning: you have more features (neurons) than samples (time bins)")
            print("Consider choosing a smaller time window for analysis")
            self.transformed_data = None
            self.coefficients = None
            self.explained_variance = None
        else:
            pca = PCA()
            scaler = StandardScaler()
            # time x neurons = samples x features
            self.zscore_matrix = scaler.fit_transform(matrix_df)
            self.scaler = scaler
            pca.fit(self.zscore_matrix)
            self.coefficients = pca.components_
            self.explained_variance = pca.explained_variance_ratio_
            self.get_cumulative_variance()
            self.make_overview_dataframe(matrix_df, event_count)
            self.full_projection = pca.transform(self.zscore_matrix)
            if condition_dict is not None:
                self.condition_pca(condition_dict)
            else:
                self.transformed_data = self.full_projection

    def make_overview_dataframe(self, matrix_df, event_count):
        column_counts = pd.DataFrame(matrix_df.columns.value_counts()).reset_index()
        column_counts.columns = ["Recording", "Number of Neurons"]

        # Add column for each event type
        for event in self.events:
            event_counts = []
            for recording in column_counts["Recording"]:
                count = event_count[recording].get(event, 0)  # get count or 0 if event not present
                event_counts.append(count)
            column_counts[f"Number of {event} events"] = event_counts

        # Add total events column
        self.recording_overview = column_counts

    def get_cumulative_variance(self):
        if self.explained_variance is not None:
            self.cumulative_variance = np.cumsum(self.explained_variance)
        else:
            self.cumulative_variance = None

    def condition_pca(self, condition_dict):
        # W = coefficients [n_PCs, n_units]
        # correct per-condition projection: subset_data @ W[:, unit_idx].T
        # = [timebins, n_i] @ [n_i, n_PCs] = [timebins, n_PCs]
        W = self.coefficients
        recording_list = np.array(self.matrix_df.columns.to_list())
        zscore_df = pd.DataFrame(data=self.zscore_matrix, columns=recording_list)
        transformed_data = {}
        for condition, rois in condition_dict.items():
            rois = [r for r in rois if r in self.recordings]
            unit_idx = np.where(np.isin(recording_list, rois))[0]
            subset_data = zscore_df.iloc[:, unit_idx].values   # [timebins, n_i]
            subset_coeff = W[:, unit_idx].T                    # [n_i, n_PCs]
            transformed_data[condition] = np.dot(subset_data, subset_coeff)
        self.transformed_data = transformed_data
        self.condition_dict = condition_dict

    def __str__(self):
        n_timebins = (self.event_length + self.post_window + self.pre_window) * 1000 / self.timebin
        total_neurons = self.recording_overview["Number of Neurons"].sum()
        if self.cumulative_variance is not None:
            pcs_for_90 = np.where(self.cumulative_variance >= 0.9)[0][0] + 1
        else:
            pcs_for_90 = None
        return (
            f"PCA Result with:\n"
            f"Events: {', '.join(self.events)}\n"
            f"Timebins per event: {n_timebins}\n"
            f"Total neurons: {total_neurons}\n"
            f"Number of recordings: {len(self.recordings)}\n"
            f"Number of Pcs needed to explain 90% of variance {pcs_for_90}"
        )

    def __repr__(self):
        return f"{self.recording_overview}"


def avg_trajectories_pca(
    spike_collection,
    event_length,
    pre_window,
    post_window=0,
    events=None,
    min_neurons=0,
    plot=True,
    d=2,
    azim=30,
    elev=20,
):
    """
    calculates a PCA matrix where each data point represents a timebin.
    PCA space is calculated from a matrix of all units and all timebins
    from every type of event in event dict or events in events.
    PCA_key is a numpy array of strings, whose index correlates with event
    type for that data point of the same index for all PCs in the pca_matrix
    pca_matrix is assigned to self.pca_matrix and the key is assigned
    as self.PCA_key for PCA plots. if save, PCA matrix is saved a dataframe wher the key is the
    row names

    Args (5 total, 2 required):
        event_length: int, length (s) of event transformed by PCA
        pre_window: int, length (s) of time prior to event onset included in PCA
        post_window: int, default=0, length(s) of time after event_length (s) included in PCA
        save: Boolean, default=False, if True, saves dataframe to collection attribute PCA_matrices
        events: list of str, default=None, event types for PCA to be applied on their firing
            rate averages, if no list given, PCA is applied on all event types in event_dict

    Returns:
        none

    """
    pc_result = avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events, min_neurons)
    if plot:
        if d == 2:
            avg_trajectory_EDA_plot(
                spike_collection, pc_result.transformed_data, pc_result.labels, event_length, pre_window, post_window
            )
        if d == 3:
            avg_trajectory_EDA_plot_3D(
                spike_collection,
                pc_result.transformed_data,
                pc_result.labels,
                event_length,
                pre_window,
                post_window,
                azim,
                elev,
            )
    return pc_result


def condition_pca(
    spike_collection,
    condition_dict,
    event_length,
    pre_window,
    post_window=0,
    events=None,
    min_neurons=0,
    plot=True,
    d=2,
    azim=30,
    elev=20,
):
    """ """
    pc_result = avg_trajectory_matrix(
        spike_collection, event_length, pre_window, post_window, events, min_neurons, condition_dict
    )
    if plot:
        if d == 2:
            condition_EDA_plot(pc_result)
        # if d == 3:
        #     condition_EDA_plot_3D(
        #         spike_collection,
        #         pc_result.transformed_data,
        #         pc_result.labels,
        #         event_length,
        #         pre_window,
        #         post_window,
        #         azim,
        #         elev,
        #     )
    return pc_result


def trial_trajectories_pca(
    spike_collection,
    event_length,
    pre_window=0,
    post_window=0,
    events=None,
    min_neurons=0,
    plot=True,
    d=2,
    azim=30,
    elev=20,
):
    pc_result = trial_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events, min_neurons)
    min_events = event_numbers(spike_collection, events, min_neurons, to_print=False)
    if plot:
        if d == 2:
            trial_trajectory_EDA_plot(
                spike_collection,
                pc_result.transformed_data,
                pc_result.labels,
                event_length,
                pre_window,
                post_window,
                min_events,
            )
        if d == 3:
            trial_trajectory_EDA_3D_plot(
                spike_collection,
                pc_result.transformed_data,
                pc_result.labels,
                event_length,
                pre_window,
                post_window,
                min_events,
                azim,
                elev,
            )
    return pc_result


def avg_trajectory_EDA_plot(spike_collection, pca_matrix, PCA_key, event_length, pre_window, post_window):
    """
    Plots PCA trajectories calculated in PCA_trajectories using the same
    pre window, post window, and event_length parameters. Each event type is
    a different color. Preevent start is signified by a square, onset of behavior
    signified by a triangle, and the end of the event is signified by a circle.
    If post-event time is included that end of post event time is signified by a diamond.
    """
    conv_factor = 1000 / spike_collection.timebin
    event_lengths = int((event_length + pre_window + post_window) * conv_factor)
    event_end = int((event_length + pre_window) * conv_factor)
    pre_window = pre_window * conv_factor
    post_window = post_window * conv_factor
    colors_dict = plt.cm.colors.CSS4_COLORS
    colors = list(colors_dict.values())
    col_counter = 10
    for i in range(0, len(PCA_key), event_lengths):
        event_label = PCA_key[i]
        onset = i if pre_window == 0 else int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + event_lengths - 1)
        plt.scatter(
            pca_matrix[i : i + event_lengths, 0],
            pca_matrix[i : i + event_lengths, 1],
            label=event_label,
            s=5,
            c=colors[col_counter],
        )
        if pre_window != 0:
            plt.scatter(pca_matrix[i, 0], pca_matrix[i, 1], marker="s", s=100, c="w", edgecolors=colors[col_counter])
            plt.scatter(pca_matrix[i, 0], pca_matrix[i, 1], marker="s", s=100, c="w", edgecolors=colors[col_counter])
        plt.scatter(
            pca_matrix[onset, 0], pca_matrix[onset, 1], marker="^", s=150, c="w", edgecolors=colors[col_counter]
        )
        plt.scatter(pca_matrix[end, 0], pca_matrix[end, 1], marker="o", s=100, c="w", edgecolors=colors[col_counter])
        if post_window != 0:
            plt.scatter(
                pca_matrix[post, 0], pca_matrix[post, 1], marker="D", s=100, c="w", edgecolors=colors[col_counter]
            )
        col_counter += 1
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    post_win_text = ""
    pre_win_text = ""
    if post_window != 0:
        post_win_text = ", Post = ◇"
    if pre_window != 0:
        pre_win_text = "Pre-event = □, "
    title = pre_win_text + "Onset = △, End = ○" + post_win_text
    plt.title(title)
    plt.show()


def trial_trajectory_EDA_plot(spike_collection, pca_matrix, PCA_key, event_length, pre_window, post_window, min_events):
    """
    Plots PCA trajectories calculated in PCA_trajectories using the same
    pre window, post window, and event_length parameters. Each event type is
    a different color. Preevent start is signified by a square, onset of behavior
    signified by a triangle, and the end of the event is signified by a circle.
    If post-event time is included that end of post event time is signified by a diamond.

    Plots individual trial PCA trajectories with each event type in a different color.
    All trials for the same event share the same color with transparency.

    Args:
        spike_collection: SpikeCollection object containing recording data
        pca_matrix: Matrix of PCA transformed data
        PCA_key: List of event labels for each point
        event_length: Length of event in seconds
        pre_window: Time before event in seconds
        post_window: Time after event in seconds
        alpha: Transparency level for trial trajectories (default=0.3)
        marker_size: Size of trajectory points (default=3)
        highlight_markers: Whether to show event markers (default=True)
    """
    conv_factor = 1000 / spike_collection.timebin
    timebins_per_trial = int((event_length + pre_window + post_window) * conv_factor)
    event_end = int((event_length + pre_window) * conv_factor)
    pre_window = pre_window * conv_factor
    post_window = post_window * conv_factor
    alpha = 0.5
    marker_size = 5
    highlight_markers = True
    # Get unique events and assign colors
    unique_events = list(set(PCA_key))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    color_dict = dict(zip(unique_events, colors))

    # Plot each trial
    for i in range(0, len(PCA_key), timebins_per_trial):
        event_label = PCA_key[i]
        color = color_dict[event_label]

        # Calculate marker positions for this trial
        onset = i if pre_window == 0 else int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + timebins_per_trial - 1)

        # Plot trajectory
        plt.plot(
            pca_matrix[i : i + timebins_per_trial, 0],
            pca_matrix[i : i + timebins_per_trial, 1],
            color=color,
            alpha=alpha,
            linewidth=0.5,
        )

        plt.scatter(
            pca_matrix[i : i + timebins_per_trial, 0],
            pca_matrix[i : i + timebins_per_trial, 1],
            s=marker_size,
            color=color,
            alpha=alpha,
        )

        # Add event markers if requested
        if highlight_markers:
            marker_kwargs = dict(s=30, alpha=1, edgecolors=color, facecolors="none")

            # Start marker
            if pre_window != 0:
                plt.scatter(pca_matrix[i, 0], pca_matrix[i, 1], marker="s", **marker_kwargs)

            # Event onset marker
            plt.scatter(pca_matrix[onset, 0], pca_matrix[onset, 1], marker="^", **marker_kwargs)

            # Event end marker
            plt.scatter(pca_matrix[end, 0], pca_matrix[end, 1], marker="o", **marker_kwargs)

            # Post-event marker if applicable
            if post_window != 0:
                plt.scatter(pca_matrix[post, 0], pca_matrix[post, 1], marker="D", **marker_kwargs)

    # Add legend with one entry per event type
    handles = [
        plt.Line2D([0], [0], color=color_dict[event], label=event, alpha=0.8, marker="o", markersize=5)
        for event in unique_events
    ]
    plt.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    # Set title based on whether post-window exists
    post_win_text = ""
    pre_win_text = ""
    if post_window != 0:
        post_win_text = ", Post = ◇"
    if pre_window != 0:
        pre_win_text = "Pre-event = □, "
    title = pre_win_text + "Onset = △, End = ○" + post_win_text
    plt.title(title)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.tight_layout()
    plt.show()


def condition_EDA_plot(pca_result):
    event_length = pca_result.event_length
    pre_window = pca_result.pre_window
    post_window = pca_result.post_window
    condition_dict = pca_result.condition_dict
    PCA_key = pca_result.labels
    conv_factor = 1000 / pca_result.timebin
    pca_matrix = pca_result.transformed_data
    event_lengths = int((event_length + pre_window + post_window) * conv_factor)
    event_end = int((event_length + pre_window) * conv_factor)
    pre_window = pre_window * conv_factor
    post_window = post_window * conv_factor
    colors_dict = plt.cm.colors.CSS4_COLORS
    colors = list(colors_dict.values())
    col_counter = 10
    for condition in condition_dict.keys():
        for i in range(0, len(PCA_key), event_lengths):
            event_label = PCA_key[i]
            onset = i if pre_window == 0 else int(i + pre_window - 1)
            end = int(i + event_end - 1)
            post = int(i + event_lengths - 1)
            plt.scatter(
                pca_matrix[condition][i : i + event_lengths, 0],
                pca_matrix[condition][i : i + event_lengths, 1],
                label=f"{condition} {event_label}",
                s=5,
                c=colors[col_counter],
            )
            if pre_window != 0:
                plt.scatter(
                    pca_matrix[condition][i, 0],
                    pca_matrix[condition][i, 1],
                    marker="s",
                    s=100,
                    c="w",
                    edgecolors=colors[col_counter],
                )
                plt.scatter(
                    pca_matrix[condition][i, 0],
                    pca_matrix[condition][i, 1],
                    marker="s",
                    s=100,
                    c="w",
                    edgecolors=colors[col_counter],
                )
            plt.scatter(
                pca_matrix[condition][onset, 0],
                pca_matrix[condition][onset, 1],
                marker="^",
                s=150,
                c="w",
                edgecolors=colors[col_counter],
            )
            plt.scatter(
                pca_matrix[condition][end, 0],
                pca_matrix[condition][end, 1],
                marker="o",
                s=100,
                c="w",
                edgecolors=colors[col_counter],
            )
            if post_window != 0:
                plt.scatter(
                    pca_matrix[condition][post, 0],
                    pca_matrix[condition][post, 1],
                    marker="D",
                    s=100,
                    c="w",
                    edgecolors=colors[col_counter],
                )
            col_counter += 1
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    post_win_text = ""
    pre_win_text = ""
    if post_window != 0:
        post_win_text = ", Post = ◇"
    if pre_window != 0:
        pre_win_text = "Pre-event = □, "
    title = pre_win_text + "Onset = △, End = ○" + post_win_text
    plt.title(title)
    plt.show()


def trial_trajectory_EDA_3D_plot(
    spike_collection, pca_matrix, PCA_key, event_length, pre_window, post_window, min_events, azim=45, elev=30
):
    """
    Plots individual trial PCA trajectories in 3D with each event type in a different color.
    All trials for the same event share the same color with transparency.

    Args:
        spike_collection: SpikeCollection object containing recording data
        pca_matrix: Matrix of PCA transformed data
        PCA_key: List of event labels for each point
        event_length: Length of event in seconds
        pre_window: Time before event in seconds
        post_window: Time after event in seconds
        alpha: Transparency level for trial trajectories (default=0.3)
        marker_size: Size of trajectory points (default=3)
        highlight_markers: Whether to show event markers (default=True)
        azim: Azimuthal viewing angle (default=45)
        elev: Elevation viewing angle (default=30)
    """
    conv_factor = 1000 / spike_collection.timebin
    timebins_per_trial = int((event_length + pre_window + post_window) * conv_factor)
    event_end = int((event_length + pre_window) * conv_factor)
    pre_window = pre_window * conv_factor
    post_window = post_window * conv_factor
    alpha = 0.5
    marker_size = 5
    highlight_markers = True

    # Get unique events and assign base colors
    unique_events = list(set(PCA_key))
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    color_dict = dict(zip(unique_events, base_colors))

    # Create 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Count trials per event for color gradient

    # Range from lighter to darker

    # Plot each trial
    event_trial_counters = {event: 0 for event in unique_events}

    for i in range(0, len(PCA_key), timebins_per_trial):
        event_label = PCA_key[i]
        base_color = color_dict[event_label]
        darkening_factor = np.linspace(0.3, 1.0, min_events[event_label])
        # Get current trial number for this event and increment counter
        trial_num = event_trial_counters[event_label]
        event_trial_counters[event_label] += 1

        # Create darker version of the color for this trial
        color = base_color * darkening_factor[trial_num]
        # Ensure alpha channel remains unchanged
        color[3] = base_color[3]

        # Calculate marker positions for this trial
        onset = i if pre_window == 0 else int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + timebins_per_trial - 1)

        # Plot trajectory
        ax.plot(
            pca_matrix[i : i + timebins_per_trial, 0],
            pca_matrix[i : i + timebins_per_trial, 1],
            pca_matrix[i : i + timebins_per_trial, 2],
            color=color,
            alpha=alpha,
            linewidth=0.8,
        )

        ax.scatter(
            pca_matrix[i : i + timebins_per_trial, 0],
            pca_matrix[i : i + timebins_per_trial, 1],
            pca_matrix[i : i + timebins_per_trial, 2],
            s=marker_size,
            color=color,
            alpha=alpha,
        )

        # Add event markers if requested
        if highlight_markers:
            marker_kwargs = dict(s=30, alpha=1, edgecolors=color, facecolors="none")

            # Start marker
            if pre_window != 0:
                ax.scatter(pca_matrix[i, 0], pca_matrix[i, 1], pca_matrix[i, 2], marker="s", **marker_kwargs)

            # Event onset marker
            ax.scatter(pca_matrix[onset, 0], pca_matrix[onset, 1], pca_matrix[onset, 2], marker="^", **marker_kwargs)

            # Event end marker
            ax.scatter(pca_matrix[end, 0], pca_matrix[end, 1], pca_matrix[end, 2], marker="o", **marker_kwargs)

            # Post-event marker if applicable
            if post_window != 0:
                ax.scatter(pca_matrix[post, 0], pca_matrix[post, 1], pca_matrix[post, 2], marker="D", **marker_kwargs)

    # Add legend with one entry per event type (using base colors)
    handles = [
        plt.Line2D([0], [0], color=color_dict[event], label=event, alpha=0.8, marker="o", markersize=5)
        for event in unique_events
    ]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    # Set labels and title
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")

    post_win_text = ""
    pre_win_text = ""
    if post_window != 0:
        post_win_text = ", Post = ◇"
    if pre_window != 0:
        pre_win_text = "Pre-event = □, "
    title = pre_win_text + "Onset = △, End = ○" + post_win_text
    plt.title(title)
    ax.view_init(azim=azim, elev=elev)

    plt.tight_layout()
    plt.show()


def avg_trajectory_EDA_plot_3D(
    spike_collection, pca_matrix, PCA_key, event_length, pre_window, post_window, azim=30, elev=50
):
    """
    Plots PCA trajectories calculated in PCA_trajectories using the same
    pre window, post window, and event_length parameters. Each event type is
    a different color. Preevent start is signified by a square, onset of behavior
    signified by a triangle, and the end of the event is signified by a circle.
    If post-event time is included that end of post event time is signified by a diamond.

    Args:
        none

    Returns:
        none
    """
    conv_factor = 1000 / spike_collection.timebin
    event_lengths = int((event_length + pre_window + post_window) * conv_factor)
    event_end = int((event_length + pre_window) * conv_factor)
    pre_window = pre_window * conv_factor
    post_window = post_window * conv_factor
    colors_dict = plt.cm.colors.CSS4_COLORS
    colors = list(colors_dict.values())
    col_counter = 10
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    for i in range(0, len(PCA_key), event_lengths):
        event_label = PCA_key[i]
        onset = i if pre_window == 0 else int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + event_lengths - 1)
        ax.scatter(
            pca_matrix[i : i + event_lengths, 0],
            pca_matrix[i : i + event_lengths, 1],
            pca_matrix[i : i + event_lengths, 2],
            label=event_label,
            s=5,
            c=colors[col_counter],
        )
        if pre_window != 0:
            ax.scatter(
                pca_matrix[i, 0],
                pca_matrix[i, 1],
                pca_matrix[i, 2],
                marker="s",
                s=100,
                c="w",
                edgecolors=colors[col_counter],
            )
        ax.scatter(
            pca_matrix[onset, 0],
            pca_matrix[onset, 1],
            pca_matrix[onset, 2],
            marker="^",
            s=150,
            c="w",
            edgecolors=colors[col_counter],
        )
        ax.scatter(
            pca_matrix[end, 0],
            pca_matrix[end, 1],
            pca_matrix[end, 2],
            marker="o",
            s=100,
            c="w",
            edgecolors=colors[col_counter],
        )
        if post_window != 0:
            ax.scatter(
                pca_matrix[post, 0],
                pca_matrix[post, 1],
                pca_matrix[post, 2],
                marker="D",
                s=100,
                c="w",
                edgecolors=colors[col_counter],
            )
        col_counter += 1
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    ax.view_init(azim=azim, elev=elev)
    post_win_text = ""
    pre_win_text = ""
    if post_window != 0:
        post_win_text = ", Post = ◇"
    if pre_window != 0:
        pre_win_text = "Pre-event = □, "
    title = pre_win_text + "Onset = △, End = ○" + post_win_text
    plt.title(title)
    plt.show()


def participation_ratio(
    spike_collection,
    event_length,
    pre_window,
    post_window=0,
    events=None,
    min_neurons=0,
    condition_dict=None,
    LOO=True,
):
    """Compute the participation ratio (PR) of the PCA eigenspectrum.

    PR = (Σλᵢ)² / Σ(λᵢ²), where λᵢ are the PCA explained variance ratios.
    A higher PR indicates variance is spread across more dimensions.

    Args:
        spike_collection: SpikeCollection or list of SpikeRecording
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, default=0, seconds after event offset
        events: list of str or None
        min_neurons: int, default=0
        condition_dict: dict or None, passed through to avg_trajectory_matrix
        LOO: bool, default=True
            If True, performs leave-one-recording-out PCA: for each recording,
            fits a fresh PCA on the remaining n-1 recordings and computes PR.
            Returns a DataFrame with n rows indexed by the left-out recording name.
            If False, fits a single PCA on all recordings and returns a one-row
            DataFrame with the PR for the full dataset.

    Returns:
        pd.DataFrame with column 'participation_ratio'.
            LOO=False: one row (no meaningful index).
            LOO=True:  n rows indexed by left-out recording name.
    """

    def _pr(evr):
        return (evr.sum() ** 2) / (evr ** 2).sum()

    if not LOO:
        pc_result = avg_trajectory_matrix(
            spike_collection, event_length, pre_window, post_window, events, min_neurons, condition_dict
        )
        if pc_result is None or pc_result.explained_variance is None:
            return None
        return pd.DataFrame({"participation_ratio": [_pr(pc_result.explained_variance)]})

    # LOO=True: leave one recording out per iteration
    if hasattr(spike_collection, "recordings"):
        recordings = spike_collection.recordings
    elif isinstance(spike_collection, list):
        recordings = spike_collection
    else:
        recordings = [spike_collection]

    recordings = [r for r in recordings if r.analyzed_neurons >= min_neurons]

    rows = []
    for i, rec in enumerate(recordings):
        loo_recs = [r for j, r in enumerate(recordings) if j != i]
        pc_result = avg_trajectory_matrix(
            loo_recs, event_length, pre_window, post_window, events, min_neurons, condition_dict
        )
        if pc_result is None or pc_result.explained_variance is None:
            continue
        rows.append({"recording": rec.name, "participation_ratio": _pr(pc_result.explained_variance)})

    return pd.DataFrame(rows).set_index("recording")


def LOO_PCA(
    spike_collection,
    event_length,
    pre_window,
    # percent_var,
    post_window=0,
    events=None,
    min_neurons=0,
    condition_dict=None,
    plot=False,
):
    pc_result_list = []
    recordings = []
    for recording in spike_collection.recordings:
        recordings.append(recording)
    for i in range(len(recordings)):
        temp_recs = recordings.copy()
        temp_recs.pop(i)
        if plot:
            print(recordings[i].name)
        if condition_dict is not None:
            pc_result = condition_pca(
                temp_recs, condition_dict, event_length, pre_window, post_window, events, min_neurons, plot
            )
        else:
            pc_result = avg_trajectories_pca(
                temp_recs, event_length, pre_window, post_window, events, min_neurons, plot
            )
        pc_result_list.append(pc_result)
    # no_PCs = PCs_needed(explained_variance_ratios, percent_var)
    # event_trajectories = event_slice(transformed_subsets, key, no_PCs, mode="multisession")
    # pairwise_distances = euclidean_distances(event_trajectories, mode="multisession")
    return pc_result_list


def average_trajectory_distances(
    spike_collection,
    event_length,
    pre_window,
    percent_var=None,
    post_window=0,
    events=None,
    min_neurons=0,
    weighted=True,
    method="recording",
):
    """Compute pairwise euclidean distances between event trajectories in PCA space.

    Both methods fit one global PCA on all n recordings (same fixed embedding), so
    distances are always comparable across recordings.

    Args:
        spike_collection: SpikeCollection or list of SpikeRecording
        event_length: float, seconds
        pre_window: float, seconds before event onset
        percent_var: float or None — variance threshold for PC selection (e.g. 0.9).
            If None, all PCs are used.
        post_window: float, default=0
        events: list of str or None
        min_neurons: int, default=0
        weighted: bool, default=True — weight distances by explained variance ratios
        method: str, default='recording'
            'recording' — fit global PCA on all n recordings; project each recording
                          individually via its own neuron columns (condition_pca).
                          n=recordings, each sample = one recording's projection.
            'LOO'       — fit global PCA on all n recordings (same fixed W); for each
                          held-out recording i, project the combined n-1 other recordings
                          via their neuron columns. Because neuron columns are
                          non-overlapping, P_LOO_i = P_full - P_i (sum of all projections
                          minus recording i's contribution). n=recordings, each sample =
                          combined projection of n-1 recordings. PCA embedding is
                          identical across all folds.

    Returns:
        pd.DataFrame — rows indexed by recording name, columns are event pairs
    """
    all_distances_df = pd.DataFrame()

    if hasattr(spike_collection, "recordings"):
        recordings = spike_collection.recordings
    elif isinstance(spike_collection, list):
        recordings = spike_collection
    else:
        recordings = [spike_collection]

    resolved_events = events if events is not None else list(recordings[0].event_dict.keys())

    # Both methods use the same global PCA fit
    condition_dict = {rec.name: [rec.name] for rec in recordings}
    pc_result = avg_trajectory_matrix(
        spike_collection, event_length, pre_window, post_window, resolved_events, min_neurons,
        condition_dict=condition_dict,
    )
    if pc_result is None or pc_result.transformed_data is None:
        return all_distances_df

    key = pc_result.labels
    ex_var = pc_result.explained_variance
    no_pcs = len(ex_var) if percent_var is None else PCs_needed(ex_var, percent_var)
    evr = ex_var[:no_pcs] if weighted else None

    if method == "recording":
        for rec_name in pc_result.recordings:
            t_mat = pc_result.transformed_data[rec_name]
            event_trajectories = event_slice(t_mat, key, no_pcs)
            recording_df = euclidean_distances(event_trajectories, recording_name=rec_name, evr=evr)
            all_distances_df = pd.concat([all_distances_df, recording_df])

    elif method == "LOO":
        # P_full = sum of all individual condition_pca projections
        all_proj = list(pc_result.transformed_data.values())
        P_full = np.sum(all_proj, axis=0)  # [T x n_PCs]
        for rec_name, P_i in pc_result.transformed_data.items():
            P_loo = P_full - P_i  # combined projection of all recordings except rec_name
            event_trajectories = event_slice(P_loo, key, no_pcs)
            recording_df = euclidean_distances(event_trajectories, recording_name=rec_name, evr=evr)
            all_distances_df = pd.concat([all_distances_df, recording_df])

    else:
        raise ValueError(f"method must be 'recording' or 'LOO', got '{method}'")

    return all_distances_df


def trajectory_length(pca_matrix, key, evr=None):
    """
    Calculates the path length of each event trajectory in PC space by summing
    successive timebin distances.

    Args (2 required, 3 total):
        pca_matrix: np.array, shape [timebins x PCs]
        key: array-like of str, event label per timebin
        evr: np.array, optional explained variance ratios used as PC weights.
            If provided, each step distance is weighted: d = sqrt(Σ evr_i * (x_i - y_i)²).
            Should match the number of PCs in pca_matrix.

    Returns:
        list of [trajectory_lengths, event_order]
            trajectory_lengths: list of float, one total path length per event
            event_order: list of str, event label for each trajectory
    """
    trajectory_lengths = []
    event_order = []
    unique_values, counts = np.unique(key, return_counts=True)
    event_len = counts[0]
    for j in range(0, len(key), event_len):
        traj_len = 0
        for i in range(event_len - 1):
            traj_len = traj_len + euclidean(pca_matrix[j + i, :], pca_matrix[j + i + 1, :], w=evr)
        trajectory_lengths.append(traj_len)
        event_order.append(key[j])
    return [trajectory_lengths, event_order]


def avg_traj_len(spike_collection, event_length, pre_window, post_window=0, events=None, min_neurons=0, percent_var=None, weighted=True, global_pca=True):
    """
    Computes the trajectory length in PCA space for each recording and event.

    Args (2 required, 9 total):
        spike_collection: SpikeCollection or list of SpikeRecording
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, default=0, seconds after event offset
        events: list of str, default=None, event types to include
        min_neurons: int, default=0, minimum neurons for a recording to be included
        percent_var: float, default=None, variance threshold for selecting number of PCs.
            If None, all PCs are used.
        weighted: bool, default=True, if True weights each PC dimension by its explained
            variance ratio: d = sqrt(Σ evr_i * (x_i - y_i)²)
        global_pca: bool, default=True, if True fits a shared PCA subspace across all
            recordings and projects each recording individually into that space (condition_pca).
            If False, fits and projects PCA independently per recording.

    Returns:
        df: DataFrame, rows are recordings, columns are event types, values are trajectory lengths
    """
    all_lengths = []

    if global_pca:
        recordings = spike_collection.recordings if hasattr(spike_collection, "recordings") else spike_collection
        condition_dict = {rec.name: [rec.name] for rec in recordings}
        pc_result = avg_trajectory_matrix(
            spike_collection, event_length, pre_window, post_window, events, min_neurons, condition_dict=condition_dict
        )
        if pc_result:
            key = pc_result.labels
            ex_var = pc_result.explained_variance
            no_pcs = len(ex_var) if percent_var is None else PCs_needed(ex_var, percent_var)
            evr = ex_var[:no_pcs] if weighted else None
            for rec_name in pc_result.recordings:
                t_mat = pc_result.transformed_data[rec_name]
                lengths, event_order = trajectory_length(t_mat[:, :no_pcs], key, evr=evr)
                row = {"recording": rec_name}
                for event, length in zip(event_order, lengths):
                    row[event] = length
                all_lengths.append(row)
    else:
        for recording in spike_collection.recordings:
            pc_result = avg_trajectory_matrix(
                recording,
                event_length,
                pre_window=pre_window,
                post_window=post_window,
                events=events,
                min_neurons=min_neurons,
            )
            if pc_result:
                t_mat = pc_result.transformed_data
                key = pc_result.labels
                ex_var = pc_result.explained_variance
                no_pcs = t_mat.shape[-1] if percent_var is None else PCs_needed(ex_var, percent_var)
                evr = ex_var[:no_pcs] if weighted else None
                lengths, event_order = trajectory_length(t_mat[:, :no_pcs], key, evr=evr)
                row = {"recording": recording.name}
                for event, length in zip(event_order, lengths):
                    row[event] = length
                all_lengths.append(row)

    return pd.DataFrame(all_lengths).set_index("recording")


def _rsa_core(matrix1, matrix2, metric="correlation"):
    """Core RSA computation between two (T x N) population matrices.

    Computes a pairwise distance vector (RDM) for each matrix across timepoints,
    then returns the Spearman correlation between the two RDMs.

    Args:
        matrix1, matrix2: np.ndarray of shape (T, N) — T timepoints, N neurons
        metric: distance metric passed to scipy pdist

    Returns:
        rsa_score: float, Spearman r
        pval: float, analytic two-tailed p-value (assumes RDM element independence —
              use permutation testing instead for valid inference)
    """
    rdm1 = pdist(matrix1, metric=metric)
    rdm2 = pdist(matrix2, metric=metric)
    rsa_score, pval = spearmanr(rdm1, rdm2)
    return rsa_score, pval


def _permutation_pval(matrix1, matrix2, observed_rho, n_perm, metric):
    """Permutation p-value for a single matrix pair.

    Shuffles timepoint rows of matrix1 n_perm times, recomputes the RDM,
    and recomputes Spearman r with matrix2's RDM to build a null distribution.
    p = proportion of null rhos >= observed_rho (one-tailed).
    """
    rdm2 = pdist(matrix2, metric=metric)
    null_rhos = np.empty(n_perm)
    for i in range(n_perm):
        perm_rdm1 = pdist(matrix1[np.random.permutation(matrix1.shape[0])], metric=metric)
        null_rhos[i], _ = spearmanr(perm_rdm1, rdm2)
    return np.mean(null_rhos >= observed_rho)



def rsa(
    spike_collection,
    events,
    event_length,
    pre_window=0,
    post_window=0,
    dim="neuron",
    metric="euclidean",
    min_neurons=3,
    plot=True,
):
    """Representational Similarity Analysis on population firing rate matrices.

    For each subject builds a matrix of shape (n_events, X) and computes its RDM
    (pairwise distances between event rows). Returns one row per subject with a
    named column per event pair and the full RDM vector. Requires at least 3 events.

    Args:
        spike_collection: SpikeCollection, fully analyzed
        events: list of str, at least 3 event types
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, seconds after event offset
        dim: str, default 'neuron' — determines the feature dimension X:
            'neuron'  — X = n_neurons; each row is mean firing rate per neuron,
                        averaged across trials then across timebins
            'timebin' — X = n_timebins; each row is mean activity per timebin,
                        averaged across trials then across neurons
            'both'    — X = n_neurons * n_timebins; each row is the trial-averaged
                        (T, N) matrix flattened neuron-first:
                        [n1t1, n1t2, ..., n1tT, n2t1, n2t2, ..., n2tT, ...]
        metric: str, distance metric for pdist (default "euclidean")
        min_neurons: int, default 3 — minimum neurons required per recording
        plot: bool, default True — violin + jitter per event pair

    Returns:
        pd.DataFrame — one row per subject, columns:
            subject, dist_{e1}_{e2} for every event pair, full_rdm_vector
    """
    if len(events) < 3:
        raise ValueError("rsa requires at least 3 events")
    if dim not in ("neuron", "timebin", "both"):
        raise ValueError("dim must be 'neuron', 'timebin', or 'both'")

    event_pairs = list(combinations(events, 2))
    pair_cols = [f"dist_{e1}_{e2}" for e1, e2 in event_pairs]

    results = []
    for recording in spike_collection.recordings:
        if recording.analyzed_neurons < min_neurons:
            print(f"Skipping {recording.name}: {recording.analyzed_neurons} neurons < min_neurons={min_neurons}")
            continue
        event_rows = []
        skip = False
        for event in events:
            trials = recording.event_firing_rates(event, event_length, pre_window, post_window)
            if len(trials) == 0:
                skip = True
                break
            avg_trials = np.mean(trials, axis=0)  # (T, N) — mean over trials
            if dim == "neuron":
                row = np.mean(avg_trials, axis=0)   # (N,) — mean over timebins
            elif dim == "timebin":
                row = np.mean(avg_trials, axis=1)   # (T,) — mean over neurons
            else:  # both
                row = avg_trials.T.flatten()        # (N*T,) — neuron-first flatten
            event_rows.append(row)
        if skip:
            continue
        mat = np.stack(event_rows, axis=0)   # (n_events, X)
        rdm = pdist(mat, metric=metric)      # (n_pairs,)
        subject = getattr(recording, "subject", recording.name)
        row_data = {"subject": subject}
        for col, dist in zip(pair_cols, rdm):
            row_data[col] = dist
        row_data["full_rdm_vector"] = rdm
        results.append(row_data)

    df = pd.DataFrame(results)

    if plot and not df.empty:
        rdm_vectors = df["full_rdm_vector"].tolist()
        rsa_scores = [
            spearmanr(rdm1, rdm2)[0]
            for rdm1, rdm2 in combinations(rdm_vectors, 2)
        ]
        rsa_scores = np.array(rsa_scores)
        fig, ax = plt.subplots(figsize=(3, 4))
        parts = ax.violinplot(rsa_scores, positions=[0], showmedians=True, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_alpha(0.4)
        jitter = np.random.uniform(-0.05, 0.05, size=len(rsa_scores))
        ax.scatter(jitter, rsa_scores, color="black", s=20, zorder=3)
        ax.set_xticks([0])
        ax.set_xticklabels([f"dim='{dim}'"])
        ax.set_ylabel("RSA (Spearman r)")
        ax.set_title("Event-geometry RSA across subject pairs")
        plt.tight_layout()
        plt.show()

    return df


def dpca_matrix(
    spike_collection,
    event_length,
    pre_window,
    post_window=0,
    events=None,
    min_neurons=0,
):
    """Build trial-averaged (N, T, E) matrix for dPCA.

    Pools neurons across all recordings that pass check_recording, mirroring avg_trajectory_matrix.

    Args:
        spike_collection: SpikeCollection or list of SpikeRecording
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, seconds after event offset
        events: list of str — event types to include; if None uses all events in first recording
        min_neurons: int, minimum analyzed_neurons for a recording to be included

    Returns:
        R           : np.ndarray (N, T, E) — trial-averaged, mean-centered per neuron
        labels      : str — dPCA labels string for the non-neuron axes, always 'te'
        neuron_keys : list of str, recording name for each neuron row (length N)
        event_list  : list of str, event name for each event slice (length E)
    """
    if isinstance(spike_collection, col.SpikeCollection):
        recordings = spike_collection.recordings
        timebin = spike_collection.timebin
    elif isinstance(spike_collection, list):
        recordings = spike_collection
        timebin = spike_collection[0].timebin
    else:
        recordings = [spike_collection]
        timebin = spike_collection.timebin

    if events is None:
        events = list(recordings[0].event_dict.keys())

    num_points = int((event_length + pre_window + post_window) * 1000 / timebin)  # T

    valid_recordings = [
        r for r in recordings if check_recording(r, min_neurons, events, to_print=True)
    ]
    if not valid_recordings:
        return None, None, None, None

    R_list = []
    neuron_keys = []

    for recording in valid_recordings:
        n_neurons = recording.analyzed_neurons
        rec_R = np.zeros((n_neurons, num_points, len(events)))  # (N_rec, T, E)

        for e_idx, event in enumerate(events):
            trials = recording.event_firing_rates(event, event_length, pre_window, post_window)
            avg = np.mean(np.stack(trials, axis=0), axis=0)  # (T, N_rec)
            rec_R[:, :, e_idx] = avg.T                        # (N_rec, T)

        R_list.append(rec_R)
        neuron_keys.extend([recording.name] * n_neurons)

    R = np.concatenate(R_list, axis=0)  # (N_total, T, E)

    
    N = R.shape[0]
    # mean-center per neuron across all conditions and timepoints
    # R -= np.mean(R.reshape(N, -1), axis=1)[:, None, None]
    # standardize per neuron across all conditions and timepoints
    NR = StandardScaler().fit_transform(R.reshape(N, -1)).reshape(R.shape)

    return R, 'te', neuron_keys, list(events), NR


class dPCAResult:
    """Result object returned by run_dpca, analogous to PCAResult.

    Attributes
    ----------
    R            : np.ndarray (N, T, E) — mean-centered trial-averaged firing rates
    NR           : np.ndarray (N, T, E) — standardized version of R (used in fit)
    Z            : dict — dPCA components keyed by marginalization ('t', 'e', 'te')
    dpca         : fitted dPCA object
    neuron_keys  : list of str — recording name for each neuron row (length N)
    event_list   : list of str — event name for each event slice (length E)
    time         : np.ndarray (T,) — time axis in seconds relative to event onset
    timebin      : float — ms per bin
    event_length : float — seconds
    pre_window   : float — seconds
    post_window  : float — seconds
    """

    _MARG_LABELS = {"t": "time", "e": "event", "te": "mixed"}

    def __init__(
        self,
        raw_matrix,
        normalized_matrix,
        transformed_matrix,
        dpca,
        neuron_keys,
        event_list,
        timebin,
        event_length,
        pre_window,
        post_window,
    ):
        self.raw_matrix = raw_matrix                    # (N, T, E) mean-centered
        self.normalized_matrix = normalized_matrix      # (N, T, E) standardized
        self.transformed_matrix = transformed_matrix    # dict keyed by marginalization
        self.dpca = dpca
        self.neuron_keys = neuron_keys
        self.event_list = event_list
        self.timebin = timebin
        self.event_length = event_length
        self.pre_window = pre_window
        self.post_window = post_window
        self.time = np.linspace(-pre_window, event_length + post_window, raw_matrix.shape[1])
        self.explained_variance = dpca.explained_variance_ratio_
        self.get_cumulative_variance()

    def get_cumulative_variance(self):
        """Compute cumulative explained variance per marginalization, mirroring PCAResult."""
        if self.explained_variance is not None:
            self.cumulative_variance = {
                key: np.cumsum(vals) for key, vals in self.explained_variance.items()
            }
        else:
            self.cumulative_variance = None

    @property
    def n_neurons(self):
        return self.raw_matrix.shape[0]

    @property
    def n_timebins(self):
        return self.raw_matrix.shape[1]

    @property
    def n_events(self):
        return self.raw_matrix.shape[2]

    def plot_components(self):
        """Plot the first dPC for each marginalization over time."""
        marg_keys = list(self.transformed_matrix.keys())
        plt.figure(figsize=(5 * len(marg_keys), 4))
        for i, key in enumerate(marg_keys, 1):
            plt.subplot(1, len(marg_keys), i)
            label = self._MARG_LABELS.get(key, key)
            for e_idx, event_name in enumerate(self.event_list):
                if label == 'event':
                    plt.plot(self.transformed_matrix[key][0, :, e_idx], self.transformed_matrix[key][1, :, e_idx], label=event_name)
                else:
                    plt.plot(self.time, self.transformed_matrix[key][0, :, e_idx], label=event_name)
            if label == 'event':
                plt.xlabel("dPC 1")
                plt.ylabel("dPC 2")
            else:
                plt.axvline(x=0, color="k", linestyle="--", linewidth=0.8)
                plt.xlabel("time (s)")
                plt.ylabel("dPC projection")
            plt.title(f"1st {label} component")
            plt.legend(fontsize=8)
        plt.suptitle("dPCA components")
        plt.tight_layout()
        plt.show()

    def plot_components_trajectory(self):
        """Plot dPC1 vs dPC2 trajectory for each marginalization."""
        marg_keys = list(self.transformed_matrix.keys())
        plt.figure(figsize=(5 * len(marg_keys), 4))
        for i, key in enumerate(marg_keys, 1):
            plt.subplot(1, len(marg_keys), i)
            label = self._MARG_LABELS.get(key, key)
            for e_idx, event_name in enumerate(self.event_list):
                # transformed_matrix[key] shape: (n_components, T, E)
                plt.plot(
                    self.transformed_matrix[key][0, :, e_idx],
                    self.transformed_matrix[key][1, :, e_idx],
                    label=event_name,
                )
            plt.xlabel("dPC 1")
            plt.ylabel("dPC 2")
            plt.title(f"{label} component (dPC1 vs dPC2)")
            plt.legend(fontsize=8)
        plt.suptitle("dPCA component trajectories")
        plt.tight_layout()
        plt.show()

    def __str__(self):
        marg_keys = list(self.transformed_matrix.keys())
        pcs_for_90 = {
            key: int(np.where(cv >= 0.9)[0][0]) + 1 if np.any(cv >= 0.9) else None
            for key, cv in (self.cumulative_variance or {}).items()
        }
        return (
            f"dPCA Result\n"
            f"  Events           : {', '.join(self.event_list)}\n"
            f"  Neurons          : {self.n_neurons}\n"
            f"  Timebins         : {self.n_timebins}  ({self.timebin} ms/bin)\n"
            f"  Marginalizations : {marg_keys}\n"
            f"  PCs for 90% var  : {pcs_for_90}"
        )

    def __repr__(self):
        return (
            f"dPCAResult | {self.n_neurons} neurons × "
            f"{self.n_timebins} timebins × {self.n_events} events | "
            f"marginalizations: {list(self.transformed_matrix.keys())}"
        )


def pseudopopulation_pca(
    spike_collection,
    event_length,
    pre_window,
    post_window=0,
    events=None,
    mode="average",
    min_neurons=0,
    d=2,
    alpha=0.8,
    linewidth=1.5,
):
    """Fit a pseudopopulation PCA and plot individual subject trajectories in the shared space.

    The pseudopopulation matrix concatenates neurons across all subjects. PCA is fit once
    on this joint matrix. Each subject's data is then projected using only their own neuron
    columns from the global weight matrix (same approach as condition_pca).

    Subjects are plotted in distinct colors (tab10). Events are shown as gradient shades of
    the subject's base color — lightest shade for the first event, full saturation for the last.
    In trial mode, individual trials are overlaid at decreasing alpha within each event shade.

    Args:
        spike_collection: SpikeCollection or list of SpikeRecording
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, seconds after event offset
        events: list of str or None — event types; if None uses all events in first recording
        mode: "average" — fit on trial-averaged FRs, project average trajectories per subject;
              "trial"   — fit on concatenated single trials (balanced to min trial count),
                          project and plot every trial per subject
        min_neurons: int, minimum analyzed_neurons for a recording to be included
        d: int, 2 or 3 — number of PCs to plot
        alpha: float, trajectory opacity (max opacity; trial mode fades lighter trials lower)
        linewidth: float, trajectory line width

    Returns:
        PCAResult with per-subject projected trajectories in transformed_data dict
    """
    if isinstance(spike_collection, col.SpikeCollection):
        recordings = spike_collection.recordings
    elif isinstance(spike_collection, list):
        recordings = spike_collection
    else:
        recordings = [spike_collection]

    resolved_events = events if events is not None else list(recordings[0].event_dict.keys())

    valid_recordings = [
        r for r in recordings
        if check_recording(r, min_neurons, resolved_events, to_print=True)
    ]
    if not valid_recordings:
        print("No valid recordings found.")
        return None

    condition_dict = {rec.name: [rec.name] for rec in valid_recordings}

    if mode == "average":
        pc_result = avg_trajectory_matrix(
            spike_collection, event_length, pre_window, post_window,
            resolved_events, min_neurons, condition_dict=condition_dict,
        )
    elif mode == "trial":
        min_events_map = event_numbers(spike_collection, resolved_events, min_neurons)
        pc_result = pca_matrix(
            spike_collection, event_length, pre_window, post_window,
            resolved_events, mode="trial", min_neurons=min_neurons,
            min_events=min_events_map, condition_dict=condition_dict,
        )
    else:
        raise ValueError(f"mode must be 'average' or 'trial', got '{mode}'")

    if pc_result is None or pc_result.transformed_data is None:
        print("PCA failed — check that you have more timebins than neurons.")
        return pc_result

    _pseudopop_plot(pc_result, mode=mode, d=d, alpha=alpha, linewidth=linewidth)
    return pc_result


def _compute_global_avg(pc_result, unique_events, timebins_per_event, mode):
    """Extract per-event global average trajectories from the full pseudopopulation projection.

    Uses pc_result.full_projection (pca.transform on all neurons together), which is
    identical to what avg_trajectories_pca plots. For trial mode, averages across trials
    within the full projection.

    Returns:
        dict mapping event name → np.ndarray of shape [timebins_per_event, n_PCs]
    """
    full = pc_result.full_projection
    labels_arr = np.array(pc_result.labels)
    global_avg = {}
    for evt_idx, evt in enumerate(unique_events):
        if mode == "average":
            seg_start = evt_idx * timebins_per_event
            global_avg[evt] = full[seg_start : seg_start + timebins_per_event]
        else:  # trial
            prior = int(sum((labels_arr == e).sum() for e in unique_events[:evt_idx]))
            n_trials = (labels_arr == evt).sum() // timebins_per_event
            trial_segs = [
                full[prior + t * timebins_per_event : prior + (t + 1) * timebins_per_event]
                for t in range(n_trials)
            ]
            global_avg[evt] = np.mean(trial_segs, axis=0)
    return global_avg


def _pseudopop_plot(pc_result, mode, d, alpha, linewidth):
    """One subplot per subject. Global average across subjects drawn in grey on every panel.
    Events → distinct tab10 colors shared across all panels.
    Trial mode → individual trials overlaid at low alpha, per-subject average at full alpha.
    """
    conv_factor = 1000 / pc_result.timebin
    timebins_per_event = int(
        (pc_result.event_length + pc_result.pre_window + pc_result.post_window) * conv_factor
    )
    event_end_bin = int((pc_result.event_length + pc_result.pre_window) * conv_factor)
    pre_bins = int(pc_result.pre_window * conv_factor)
    post_bins = int(pc_result.post_window * conv_factor)

    unique_events = list(dict.fromkeys(pc_result.labels))
    subjects = list(pc_result.transformed_data.keys())
    n_subjects = len(subjects)
    n_events = len(unique_events)

    tab10_colors = plt.cm.tab10(np.linspace(0, 0.9, max(n_events, 1)))
    evt_color = {evt: tab10_colors[i] for i, evt in enumerate(unique_events)}

    # Global average gets a distinct dark neutral per event: black → dark grey
    dark_neutrals = np.linspace(0.0, 0.55, max(n_events, 1))
    evt_avg_color = {evt: str(dark_neutrals[i]) for i, evt in enumerate(unique_events)}

    global_avg = _compute_global_avg(pc_result, unique_events, timebins_per_event, mode)

    ncols = min(3, n_subjects)
    nrows = (n_subjects + ncols - 1) // ncols

    if d == 3:
        fig = plt.figure(figsize=(5 * ncols, 4 * nrows))
        all_axes = [
            fig.add_subplot(nrows, ncols, i + 1, projection="3d")
            for i in range(n_subjects)
        ]
    else:
        fig, axes_grid = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
        all_axes = [axes_grid[r][c] for r in range(nrows) for c in range(ncols)]
        for ax in all_axes[n_subjects:]:
            ax.set_visible(False)

    labels_arr = np.array(pc_result.labels)

    for subj_idx, subj_name in enumerate(subjects):
        ax = all_axes[subj_idx]
        traj = pc_result.transformed_data[subj_name]

        # Global average per event: distinct dark neutrals (black → dark grey), dashed
        for evt in unique_events:
            avg_seg = global_avg[evt]
            avg_color = evt_avg_color[evt]
            if d == 2:
                ax.plot(avg_seg[:, 0], avg_seg[:, 1],
                        color=avg_color, alpha=0.85, linewidth=linewidth * 1.5, linestyle="--", zorder=1)
            else:
                ax.plot(avg_seg[:, 0], avg_seg[:, 1], avg_seg[:, 2],
                        color=avg_color, alpha=0.85, linewidth=linewidth * 1.5, linestyle="--", zorder=1)

        # Subject trajectories per event
        for evt_idx, evt in enumerate(unique_events):
            color = evt_color[evt]

            if mode == "average":
                seg_start = evt_idx * timebins_per_event
                segment = traj[seg_start : seg_start + timebins_per_event]
                _plot_segment(ax, segment, color, alpha, linewidth, d,
                              pre_bins, event_end_bin, post_bins, markers=True)

            else:  # trial
                prior = int(sum((labels_arr == e).sum() for e in unique_events[:evt_idx]))
                n_trials = (labels_arr == evt).sum() // timebins_per_event
                trial_segs = []
                for trial_i in range(n_trials):
                    t_start = prior + trial_i * timebins_per_event
                    segment = traj[t_start : t_start + timebins_per_event]
                    trial_segs.append(segment)
                    _plot_segment(ax, segment, color, 0.2, linewidth * 0.5, d,
                                  pre_bins, event_end_bin, post_bins, markers=False)
                # Per-subject average for this event on top
                if trial_segs:
                    avg_seg = np.mean(trial_segs, axis=0)
                    _plot_segment(ax, avg_seg, color, alpha, linewidth, d,
                                  pre_bins, event_end_bin, post_bins, markers=True)

        ax.set_title(subj_name, fontsize=9)
        ax.set_xlabel("PC1", fontsize=8)
        ax.set_ylabel("PC2", fontsize=8)
        if d == 3:
            ax.set_zlabel("PC3", fontsize=8)

    # Shared legend on the figure
    legend_handles = [
        plt.Line2D([0], [0], color=evt_color[evt], linewidth=2, label=evt)
        for evt in unique_events
    ]
    for evt in unique_events:
        legend_handles.append(
            plt.Line2D([0], [0], color=evt_avg_color[evt], linewidth=2, linestyle="--",
                       alpha=0.85, label=f"{evt} (global avg)")
        )

    marker_text = ""
    if pre_bins > 0:
        marker_text += "Pre = □, "
    marker_text += "Onset = △, End = ○"
    if post_bins > 0:
        marker_text += ", Post = ◇"

    fig.legend(handles=legend_handles, loc="upper right", fontsize=8, framealpha=0.9)
    fig.suptitle(f"Pseudopopulation PCA ({mode}) — {marker_text}", fontsize=10, y=1.01)
    plt.tight_layout()
    plt.show()


def _plot_segment(ax, segment, color, alpha, linewidth, d, pre_bins, event_end_bin, post_bins, markers):
    """Draw a single trajectory segment and optional event markers onto ax."""
    if len(segment) == 0:
        return
    marker_kw = dict(s=60, zorder=5, edgecolors=color, facecolors="none", linewidths=1.2)
    if d == 2:
        ax.plot(segment[:, 0], segment[:, 1], color=color, alpha=alpha, linewidth=linewidth)
        if markers:
            onset = pre_bins if pre_bins > 0 else 0
            if pre_bins > 0:
                ax.scatter(segment[0, 0], segment[0, 1], marker="s", **marker_kw)
            ax.scatter(segment[onset, 0], segment[onset, 1], marker="^", **marker_kw)
            ax.scatter(
                segment[event_end_bin - 1, 0], segment[event_end_bin - 1, 1],
                marker="o", **marker_kw,
            )
            if post_bins > 0:
                ax.scatter(segment[-1, 0], segment[-1, 1], marker="D", **marker_kw)
    else:
        ax.plot(
            segment[:, 0], segment[:, 1], segment[:, 2],
            color=color, alpha=alpha, linewidth=linewidth,
        )
        if markers:
            onset = pre_bins if pre_bins > 0 else 0
            if pre_bins > 0:
                ax.scatter(
                    segment[0, 0], segment[0, 1], segment[0, 2], marker="s", **marker_kw
                )
            ax.scatter(
                segment[onset, 0], segment[onset, 1], segment[onset, 2], marker="^", **marker_kw
            )
            ax.scatter(
                segment[event_end_bin - 1, 0],
                segment[event_end_bin - 1, 1],
                segment[event_end_bin - 1, 2],
                marker="o", **marker_kw,
            )
            if post_bins > 0:
                ax.scatter(
                    segment[-1, 0], segment[-1, 1], segment[-1, 2], marker="D", **marker_kw
                )


def run_dpca(
    spike_collection,
    event_length,
    pre_window,
    post_window=0,
    events=None,
    min_neurons=0,
    protect_time=True,
    plot=True,
):
    """Fit dPCA on population firing rates organized as (N, T, E).

    Args:
        spike_collection : SpikeCollection or list of SpikeRecording
        event_length     : float, seconds
        pre_window       : float, seconds before event onset
        post_window      : float, seconds after event offset
        events           : list of str — event types; if None uses all events in first recording
        min_neurons      : int, minimum analyzed_neurons threshold
        protect_time     : bool — set dpca.protect=['t'] to protect time axis during shuffle
        plot             : bool — call result.plot_components() before returning

    Returns:
        dPCAResult object
    """
    from dPCA import dPCA as dPCA_lib

    if isinstance(spike_collection, col.SpikeCollection):
        timebin = spike_collection.timebin
    elif isinstance(spike_collection, list):
        timebin = spike_collection[0].timebin
    else:
        timebin = spike_collection.timebin

    R, labels, neuron_keys, event_list, NR = dpca_matrix(
        spike_collection, event_length, pre_window, post_window, events, min_neurons
    )
    if R is None:
        return None

    dpca = dPCA_lib.dPCA(labels=labels)
    if protect_time:
        dpca.protect = ["t"]

    Z = dpca.fit_transform(NR)

    result = dPCAResult(
        raw_matrix=R,
        normalized_matrix=NR,
        transformed_matrix=Z,
        dpca=dpca,
        neuron_keys=neuron_keys,
        event_list=event_list,
        timebin=timebin,
        event_length=event_length,
        pre_window=pre_window,
        post_window=post_window,
    )

    if plot:
        result.plot_components()

    return result


def procrustes_alignment(collection, events, event_length, pre_window, post_window, min_neurons, scaled=False, n_pcs=None, n_perm=0):
    """Compute Procrustes disparity between all pairs of subjects for each event type.

    PCA is fit independently per subject. For each subject pair and event, one trajectory
    is aligned to the other using Procrustes analysis and the residual disparity is returned.
    Because each subject's PCA axes are arbitrary, Procrustes captures differences in
    manifold geometry rather than coordinate-frame alignment.

    Args:
        collection: SpikeCollection or list of SpikeRecording
        events: list of str, event types to include
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, seconds after event offset
        min_neurons: int, minimum analyzed_neurons for a recording to be included
        scaled: bool, default=False
            If False, uses orthogonal Procrustes (rotation only, no scaling):
                R, _ = orthogonal_procrustes(mtx1, mtx2)
                disparity = ||mtx1 - mtx2 @ R||
            If True, uses full Procrustes (rotation + isotropic scaling + translation):
                mtx1, mtx2, disparity = procrustes(mtx1, mtx2)
        n_pcs: int or None, default=None
            Number of PCs to use for each trajectory. If None, uses the minimum number
            of PCs available across the subject pair.
        n_perm: int, default=0
            Number of permutations for null distribution. If > 0, shuffles timebin rows
            of mtx1 before alignment on each permutation. p-value = proportion of null
            disparities <= observed (low p = observed disparity is unusually low =
            trajectories are more aligned than chance).

    Returns:
        pd.DataFrame — index is (subject1, subject2) tuples, columns are event disparity
            values, and if n_perm > 0, additional pval_{event} columns.
    """
    if hasattr(collection, "recordings"):
        recordings = collection.recordings
    elif isinstance(collection, list):
        recordings = collection
    else:
        recordings = [collection]

    valid_recordings = [
        r for r in recordings if check_recording(r, min_neurons, events, to_print=True)
    ]
    if len(valid_recordings) < 2:
        print("Need at least 2 valid recordings for Procrustes alignment.")
        return pd.DataFrame()

    pc_results = {}
    for recording in valid_recordings:
        pc_result = avg_trajectory_matrix(
            recording, event_length, pre_window, post_window, events, min_neurons
        )
        if pc_result is not None and pc_result.transformed_data is not None:
            pc_results[recording.name] = pc_result

    if len(pc_results) < 2:
        print("Fewer than 2 recordings had valid PCA results.")
        return pd.DataFrame()

    def _disparity(m1, m2):
        if scaled:
            _, _, d = scipy_procrustes(m1, m2)
        else:
            R, _ = orthogonal_procrustes(m1, m2)
            d = np.linalg.norm(m1 - m2 @ R)
        return d

    rows = []
    for subj1, subj2 in combinations(list(pc_results.keys()), 2):
        traj1 = pc_results[subj1].transformed_data
        traj2 = pc_results[subj2].transformed_data
        key = pc_results[subj1].labels
        k = n_pcs if n_pcs is not None else min(traj1.shape[1], traj2.shape[1])

        event_trajs1 = event_slice(traj1, key, k)
        event_trajs2 = event_slice(traj2, key, k)

        row = {"subject_pair": (subj1, subj2)}
        for event in events:
            mtx1 = event_trajs1[event]
            mtx2 = event_trajs2[event]
            observed = _disparity(mtx1, mtx2)
            row[event] = observed

            if n_perm > 0:
                null = np.array([
                    _disparity(mtx1[np.random.permutation(len(mtx1))], mtx2)
                    for _ in range(n_perm)
                ])
                row[f"pval_{event}"] = np.mean(null <= observed)

        rows.append(row)

    return pd.DataFrame(rows).set_index("subject_pair")
