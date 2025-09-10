import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
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
    if recording.good_neurons < min_neurons:
        if to_print:
            print(f"Excluding {recording.name} with {recording.good_neurons} neurons")
        return False
    for event in events:
        if len(recording.event_dict[event]) == 1:
            if recording.event_dict[event][0][1] - recording.event_dict[event][0][0] == 0:
                if to_print:
                    print(f"Excluding {recording.name}, it has no {event} events")
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
