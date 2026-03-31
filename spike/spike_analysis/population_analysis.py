import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean, pdist
from scipy.stats import spearmanr, sem
from itertools import combinations
import spike.spike_analysis.spike_collection as col
import spike.spike_analysis.spike_recording
from sklearn.preprocessing import StandardScaler


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


def geodesic_distances(event_trajectories, recording_name=None):
    """
    Calculates the euclidean distances between all trajectories in the event_trajectory dictionary,

    Arguments(1 required, 2 total):
        event_trajectories: dictionary
            keys: str, event names
            values: numpy arrays of shape [session x timebins x PCs] or [timebins x PCs]
        recording_name: str, optional index labeled for the resulting dataframe

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
        dist = distance_bw_trajectories(event1, event2)
        distances.append(dist)

    # Create column names from pairs
    column_names = [f"{pair[0]}_{pair[1]}" for pair in event_pairs]
    # Create DataFrame
    if recording_name is not None:
        df = pd.DataFrame([distances], columns=column_names, index=[recording_name])

    else:
        df = pd.DataFrame([distances], columns=column_names)

    return df


def distance_bw_trajectories(trajectory1, trajectory2):
    """
    Calculates the geodesic distance between two event trajectories by summing the distance between
    congruent timebins across trajectories.

    Arugments (2 required):
        trajectory1 & trajectory2: numpy ararys of shape [session x timebin x PCs] pr [timebin x PCs]

    Returns (1):
        geodesic_distances: either a single value for 1 session's trajectories, or a list of distances across
        all sessions trajectories
    """
    if len(trajectory1.shape) == 3:
        geodesic_distances = []
        for session in range(trajectory1.shape[0]):
            dist_bw_tb = 0
            for i in range(trajectory1.shape[1]):
                dist_bw_tb = dist_bw_tb + euclidean(trajectory1[session, i, :], trajectory2[session, i, :])
            geodesic_distances.append(dist_bw_tb)
    if len(trajectory1.shape) == 2:
        dist_bw_tb = 0
        for i in range(trajectory1.shape[0]):
            dist_bw_tb = dist_bw_tb + euclidean(trajectory1[i, :], trajectory2[i, :])
        geodesic_distances = dist_bw_tb
    return geodesic_distances


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
            pca.fit(matrix_df)
            self.coefficients = pca.components_
            self.explained_variance = pca.explained_variance_ratio_
            self.get_cumulative_variance()
            self.make_overview_dataframe(matrix_df, event_count)
            if condition_dict is not None:
                self.condition_pca(condition_dict)
            else:

                self.transformed_data = pca.transform(self.zscore_matrix)

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
        coefficients = self.coefficients
        recording_list = self.matrix_df.columns.to_list()
        zscore_matrix = pd.DataFrame(data=self.zscore_matrix, columns=recording_list)
        coefficients_df = pd.DataFrame(data=coefficients, index=recording_list)
        transformed_data = {}
        # transformed data dict: conditions for keys, values is a transformed data array
        for condition, rois in condition_dict.items():
            rois = [recording for recording in rois if recording in self.recordings]
            # trim weight matrix for only those neurons in recordings of that condition
            subset_coeff = coefficients_df[coefficients_df.index.isin(rois)]
            subset_data = zscore_matrix[rois]
            condition_data = np.dot(subset_data, subset_coeff)
            # transform each condition with condition specific weight matrix
            # T (timebins x pcs) = D (timebins x neurons). W (pcs x neurons)
            transformed_data[condition] = condition_data
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
    # pairwise_distances = geodesic_distances(event_trajectories, mode="multisession")
    return pc_result_list


def avg_geo_dist(spike_collection, event_length, pre_window, percent_var, post_window=0, events=None, min_neurons=0):
    all_distances_df = pd.DataFrame()

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
            no_pcs = PCs_needed(ex_var, percent_var)
            event_trajectories = event_slice(
                t_mat,
                key,
                no_pcs,
            )

            # Get distances DataFrame for this recording
            recording_df = geodesic_distances(event_trajectories, recording_name=recording.name)

            # Concatenate with main DataFrame
            all_distances_df = pd.concat([all_distances_df, recording_df])

    return all_distances_df


def _rsa_core(matrix1, matrix2, metric="euclidean"):
    """Core RSA computation between two (T x N) population matrices.

    Computes a pairwise distance vector (RDM) for each matrix across timepoints,
    then returns the Spearman correlation between the two RDMs.

    Args:
        matrix1, matrix2: np.ndarray of shape (T, N) — T timepoints, N neurons
        metric: distance metric passed to scipy pdist

    Returns:
        rsa_score: float, Spearman r
        pval: float, two-tailed p-value
    """
    rdm1 = pdist(matrix1, metric=metric)
    rdm2 = pdist(matrix2, metric=metric)
    rsa_score, pval = spearmanr(rdm1, rdm2)
    return rsa_score, pval


def rsa(
    spike_collection,
    events,
    event_length,
    pre_window=0,
    post_window=0,
    across_subjects=False,
    across_events=False,
    metric="euclidean",
    plot=True,
):
    """Representational Similarity Analysis on population firing rate matrices.

    Args:
        spike_collection: SpikeCollection, fully analyzed
        events: list of str, event types to include
        event_length: float, seconds
        pre_window: float, seconds before event onset
        post_window: float, seconds after event offset
        across_subjects: bool — compare same event type between every pair of subjects
        across_events: bool — compare different event types within each subject
        metric: str, distance metric for pdist (default "euclidean")

    Returns:
        list of dicts, one per comparison — convertible to a DataFrame
    """
    if across_subjects and across_events:
        raise ValueError("Set either across_subjects or across_events, not both.")

    results = []

    # within subject, within event: RSA between every pair of trials, averaged per recording+event
    if not across_subjects and not across_events:
        for recording in spike_collection.recordings:
            for event in events:
                trials = recording.event_firing_rates(event, event_length, pre_window, post_window)
                if len(trials) < 2:
                    continue
                trial_rsa_scores = []
                for trial1, trial2 in combinations(trials, 2):
                    score, _ = _rsa_core(trial1, trial2, metric)
                    trial_rsa_scores.append(score)
                results.append({
                    "comparison": "within_subject_within_event",
                    "recording": recording.name,
                    "subject": getattr(recording, "subject", None),
                    "event": event,
                    "rsa": np.mean(trial_rsa_scores),
                    "n_pairs": len(trial_rsa_scores),
                })

    if across_subjects:
        for event in events:
            # build {subject: avg_matrix (T x N)} for this event
            subject_matrices = {}
            for recording in spike_collection.recordings:
                trials = recording.event_firing_rates(event, event_length, pre_window, post_window)
                if len(trials) == 0:
                    continue
                avg_matrix = np.mean(trials, axis=0)  # (T, N)
                subject_matrices[recording.subject] = avg_matrix

            for (subj1, mat1), (subj2, mat2) in combinations(subject_matrices.items(), 2):
                rsa_score, pval = _rsa_core(mat1, mat2, metric)
                results.append({
                    "comparison": "across_subjects",
                    "event": event,
                    "subject_1": subj1,
                    "subject_2": subj2,
                    "rsa": rsa_score,
                    "pval": pval,
                })

    if across_events:
        for recording in spike_collection.recordings:
            # build {event: avg_matrix (T x N)} for this recording
            event_matrices = {}
            for event in events:
                trials = recording.event_firing_rates(event, event_length, pre_window, post_window)
                if len(trials) == 0:
                    continue
                event_matrices[event] = np.mean(trials, axis=0)  # (T, N)

            for (event1, mat1), (event2, mat2) in combinations(event_matrices.items(), 2):
                rsa_score, pval = _rsa_core(mat1, mat2, metric)
                results.append({
                    "comparison": "across_events",
                    "recording": recording.name,
                    "subject": getattr(recording, "subject", None),
                    "event_1": event1,
                    "event_2": event2,
                    "rsa": rsa_score,
                    "pval": pval,
                })

    if plot and results:
        if not across_subjects and not across_events:
            # group per-subject average RSA by event, then mean/SEM across subjects
            event_rsa = {}
            for r in results:
                event_rsa.setdefault(r["event"], []).append(r["rsa"])
            labels = list(event_rsa.keys())
            means = [np.mean(event_rsa[e]) for e in labels]
            errors = [sem(event_rsa[e]) for e in labels]
            x = np.arange(len(labels))
            plt.figure(figsize=(max(4, len(labels) * 1.5), 4))
            plt.bar(x, means, yerr=errors, capsize=5, width=0.5)
            plt.xticks(x, labels)
            plt.ylabel("RSA (Spearman r)")
            plt.title("Within-subject trial-to-trial RSA per event")
            plt.tight_layout()
            plt.show()

        elif across_subjects:
            # group RSA values by event across all subject pairs
            event_rsa = {}
            for r in results:
                event_rsa.setdefault(r["event"], []).append(r["rsa"])
            labels = list(event_rsa.keys())
            means = [np.mean(event_rsa[e]) for e in labels]
            errors = [sem(event_rsa[e]) for e in labels]
            x = np.arange(len(labels))
            plt.figure(figsize=(max(4, len(labels) * 1.5), 4))
            plt.bar(x, means, yerr=errors, capsize=5, width=0.5)
            plt.xticks(x, labels)
            plt.ylabel("RSA (Spearman r)")
            plt.title("RSA across subjects — average subject pair per event")
            plt.tight_layout()
            plt.show()

        elif across_events:
            # group RSA values by event pair across all subjects
            pair_rsa = {}
            for r in results:
                pair_label = f"{r['event_1']} vs {r['event_2']}"
                pair_rsa.setdefault(pair_label, []).append(r["rsa"])
            labels = list(pair_rsa.keys())
            means = [np.mean(pair_rsa[p]) for p in labels]
            errors = [sem(pair_rsa[p]) for p in labels]
            x = np.arange(len(labels))
            plt.figure(figsize=(max(4, len(labels) * 1.5), 4))
            plt.bar(x, means, yerr=errors, capsize=5, width=0.5)
            plt.xticks(x, labels, rotation=15, ha="right")
            plt.ylabel("RSA (Spearman r)")
            plt.title("RSA across events — average across subjects per event pair")
            plt.tight_layout()
            plt.show()

    return results


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
