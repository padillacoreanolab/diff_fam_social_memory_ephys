import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from itertools import combinations


def get_indices(repeated_items_list):
    """
    Takes in an indexed key or a list of repeated items,
    creates a list of indices that correspond to each unique item.

    Args (1):
        repeated_items_list: list, list of repeated items

    Returns:
        item_indices: list of tuples, where the first element
            is the first index of an item, and the second
            element is the last index of that item
    """
    is_first = True
    item_indices = []
    for i in range(len(repeated_items_list)):
        if is_first:
            current_item = repeated_items_list[i]
            start_index = 0
            is_first = False
        else:
            if repeated_items_list[i] == current_item:
                end_index = i
                if i == (len(repeated_items_list) - 1):
                    item_indices.append([start_index, end_index])
            else:
                item_indices.append([start_index, end_index])
                start_index = i
                current_item = repeated_items_list[i]
    return item_indices


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


def event_slice(transformed_subsets, key, no_PCs, mode):
    """
    Takes in a matrix T (session x timebins x pcs) (mode = 'multisession')
    or (timebins x pcs) (mode = 'single')
    and an event key to split the matrix by event and trim it to no_PCs.

    Args (3):
        transformed_subsets: np.array, d(session X timebin X PCS)
        key: list of str, each element is an event type and
            corresponds to the timebin dimension indices of
            the transformed_subsets matrix
        no_PCs: int, number of PCs required to explain a variance threshold
        mode: {'multisession', 'single'}; multisession calculates event slices
            for many transformed subsets, single calculates event slices for a
            single session
    Returns:
        trajectories: dict, events to trajectories across
            each LOO PCA embedding
            keys: str, event types
            values: np.array, d=(session x timebins x no_PCs)
    """
    event_indices = get_indices(key)
    events = np.unique(key)
    trajectories = {}
    for i in range(len(event_indices)):
        event = events[i]
        start = event_indices[i][0]
        stop = event_indices[i][1]
        if mode == "multisession":
            event_trajectory = transformed_subsets[:, start : stop + 1, :no_PCs]
        if mode == "single":
            event_trajectory = transformed_subsets[start : stop + 1, :no_PCs]
        trajectories[event] = event_trajectory
    return trajectories


def geodesic_distances(event_trajectories, mode):
    pair_distances = {}
    for pair in list(combinations(event_trajectories.keys(), 2)):
        event1 = event_trajectories[pair[0]]
        event2 = event_trajectories[pair[1]]
        pair_distances[pair] = distance_bw_trajectories(event1, event2, mode)
    return pair_distances


def distance_bw_trajectories(trajectory1, trajectory2, mode):
    geodesic_distances = []
    if mode == "multisession":
        for session in range(trajectory1.shape[0]):
            dist_bw_tb = 0
            for i in range(trajectory1.shape[1]):
                dist_bw_tb = dist_bw_tb + euclidean(trajectory1[session, i, :], trajectory2[session, i, :])
            geodesic_distances.append(dist_bw_tb)
    if mode == "single":
        dist_bw_tb = 0
        for i in range(trajectory1.shape[0]):
            dist_bw_tb = dist_bw_tb + euclidean(trajectory1[i, :], trajectory2[i, :])
        geodesic_distances.append(dist_bw_tb)
    return geodesic_distances


def avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window=0, events=None):
    """
    Args (5 total, 2 required):
        event_length: int, length (s) of event transformed by PCA
        pre_window: int, length (s) of time prior to event onset included in PCA
        post_window: int, default=0, length(s) of time after event_length (s) included in PCA
        save: Boolean, default=False, if True, saves dataframe to collection attribute PCA_matrices
        events: list of str, default=None, event types for PCA to be applied on their firing
            rate averages, if no list given, PCA is applied on all event types in event_dict

    Returns:
        PCA_dict

    """
    is_first_recording = True
    for recording in spike_collection.collection:
        if events is None:
            events = list(recording.event_dict.keys())
            PCA_dict_key = None
        else:
            for i in range(len(events)):
                if i == 0:
                    PCA_dict_key = events[i]
                else:
                    PCA_dict_key = PCA_dict_key + events[i]
        is_first_event = True
        for event in events:
            event_firing_rates = recording.__event_firing_rates__(event, event_length, pre_window, post_window)
            event_firing_rates = np.array(event_firing_rates)
            event_averages = np.nanmean(event_firing_rates, axis=0)
            if is_first_event:
                PCA_matrix = event_averages
                PCA_event_key = [event] * int((event_length + pre_window + post_window) * 1000 / recording.timebin)
                is_first_event = False
            else:
                PCA_matrix = np.concatenate((PCA_matrix, event_averages), axis=1)
                next_key = [event] * int((event_length + pre_window + post_window) * 1000 / recording.timebin)
                PCA_event_key = np.concatenate((PCA_event_key, next_key), axis=0)
        if is_first_recording:
            PCA_master_matrix = PCA_matrix
            PCA_recording_key = [recording.name] * PCA_matrix.shape[0]
            is_first_recording = False
        else:
            PCA_master_matrix = np.concatenate((PCA_master_matrix, PCA_matrix), axis=0)
            next_recording_key = [recording.name] * PCA_matrix.shape[0]
            PCA_recording_key = np.concatenate((PCA_recording_key, next_recording_key), axis=0)
    matrix = np.transpose(PCA_master_matrix)
    matrix_df = pd.DataFrame(data=matrix, columns=PCA_recording_key, index=PCA_event_key)
    key = np.array(matrix_df.index.to_list())
    if matrix.shape[0] < matrix.shape[1]:
        print("you have more features (neurons) than samples (time bins)")
        print("please choose a smaller time window for analysis")
        return {
            "raw data": matrix_df,
            "transformed data": None,
            "labels": key,
            "coefficients": None,
            "explained variance": None,
        }
    else:
        pca = PCA()
        pca.fit(matrix_df)
        transformed_matrix = pca.transform(matrix_df)
        coefficients = pca.components_
        exp_var_ratios = pca.explained_variance_ratio_
        return {
            "raw data": matrix_df,
            "transformed data": transformed_matrix,
            "labels": key,
            "coefficients": coefficients,
            "explained variance": exp_var_ratios,
        }


def PCA_avg_trajectories(
    spike_collection, event_length, pre_window, post_window=0, events=None, plot=True, d=2, azim=30, elev=20
):
    """
    calculates a PCA matrix where each data point represents a timebin.
    PCA space is calculated from a matrix of all units and all timebins
    from every type of event in event dict or events in events.
    PCA_key is a numpy array of strings, whose index correlates with event
    type for that data point of the same index for all PCs in the PCA_matrix
    PCA_matrix is assigned to self.PCA_matrix and the key is assigned
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
    pc_dict = avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events)
    transformed_matrix = pc_dict["transformed data"]
    if events is not None:
        for i in range(len(events)):
            if i == 0:
                PCA_dict_key = events[i]
            else:
                PCA_dict_key = PCA_dict_key + events[i]
    PCA_event_key = pc_dict["labels"]
    if plot:
        if d == 2:
            PCA_EDA_plot(transformed_matrix, PCA_event_key, event_length, pre_window, post_window)
        if d == 3:
            PCA_EDA_plot_3D(transformed_matrix, PCA_event_key, event_length, pre_window, post_window, azim, elev)
    return pc_dict


def PCA_EDA_plot(spike_collection, PCA_matrix, PCA_key, event_length, pre_window, post_window):
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
        onset = int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + event_lengths - 1)
        plt.scatter(
            PCA_matrix[i : i + event_lengths, 0],
            PCA_matrix[i : i + event_lengths, 1],
            label=event_label,
            s=5,
            c=colors[col_counter],
        )
        plt.scatter(PCA_matrix[i, 0], PCA_matrix[i, 1], marker="s", s=100, c="w", edgecolors=colors[col_counter])
        plt.scatter(
            PCA_matrix[onset, 0], PCA_matrix[onset, 1], marker="^", s=150, c="w", edgecolors=colors[col_counter]
        )
        plt.scatter(PCA_matrix[end, 0], PCA_matrix[end, 1], marker="o", s=100, c="w", edgecolors=colors[col_counter])
        if post_window != 0:
            plt.scatter(
                PCA_matrix[post, 0], PCA_matrix[post, 1], marker="D", s=100, c="w", edgecolors=colors[col_counter]
            )
        col_counter += 1
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    if post_window != 0:
        plt.title("Preevent = square, Onset = triangle, End of event = circle, Post event = Diamond")
    else:
        plt.title("Preevent = square, Onset = triangle, End of event = circle")
    plt.show()


def PCA_EDA_plot_3D(spike_collection, PCA_matrix, PCA_key, event_length, pre_window, post_window, azim=30, elev=50):
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
        onset = int(i + pre_window - 1)
        end = int(i + event_end - 1)
        post = int(i + event_lengths - 1)
        ax.scatter(
            PCA_matrix[i : i + event_lengths, 0],
            PCA_matrix[i : i + event_lengths, 1],
            PCA_matrix[i : i + event_lengths, 2],
            label=event_label,
            s=5,
            c=colors[col_counter],
        )
        ax.scatter(
            PCA_matrix[i, 0],
            PCA_matrix[i, 1],
            PCA_matrix[i, 2],
            marker="s",
            s=100,
            c="w",
            edgecolors=colors[col_counter],
        )
        ax.scatter(
            PCA_matrix[onset, 0],
            PCA_matrix[onset, 1],
            PCA_matrix[onset, 2],
            marker="^",
            s=150,
            c="w",
            edgecolors=colors[col_counter],
        )
        ax.scatter(
            PCA_matrix[end, 0],
            PCA_matrix[end, 1],
            PCA_matrix[end, 2],
            marker="o",
            s=100,
            c="w",
            edgecolors=colors[col_counter],
        )
        if post_window != 0:
            ax.scatter(
                PCA_matrix[post, 0],
                PCA_matrix[post, 1],
                PCA_matrix[post, 2],
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
    if post_window != 0:
        ax.set_title("Preevent = square, Onset = triangle, End of event = circle, Post event = Diamond")
    else:
        ax.set_title("Preevent = square, Onset = triangle, End of event = circle")
    plt.show()


def LOO_PCA(spike_collection, event_length, pre_window, percent_var, post_window=0, events=None):
    pc_dict = avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events)
    full_PCA_matrix = pc_dict["raw data"]
    key = pc_dict["labels"]
    coefficients = pc_dict["coefficients"]
    explained_variance_ratios = pc_dict["explained variance"]
    transformed_subsets = []
    i = 0
    recording_indices = get_indices(full_PCA_matrix.columns.to_list())
    for i in range(len(recording_indices)):
        start = recording_indices[i][0]
        stop = recording_indices[i][1]
        subset_df = full_PCA_matrix.drop(full_PCA_matrix.columns[start:stop], axis=1)
        subset_array = subset_df.values
        subset_coeff = np.delete(coefficients, np.s_[start : stop + 1], axis=0)
        transformed_subset = np.dot(subset_array, subset_coeff)
        transformed_subsets.append(transformed_subset)
    transformed_subsets = np.stack(transformed_subsets, axis=0)
    no_PCs = PCs_needed(explained_variance_ratios, percent_var)
    event_trajectories = event_slice(transformed_subsets, key, no_PCs, mode="multisession")
    pairwise_distances = geodesic_distances(event_trajectories, mode="multisession")
    return pairwise_distances


# hmm think how this works for one recording
def avg_geo_dist(spike_collection, event_length, pre_window, percent_var, post_window=0, events=None):
    temp_pairwise_distances = {}
    is_first = True
    for recording in spike_collection.collection:
        pc_dict = avg_trajectory_matrix(event_length, pre_window, post_window, events)
        t_mat = pc_dict["transformed data"]
        key = pc_dict["labels"]
        ex_var = pc_dict["explained variance"]
        no_pcs = PCs_needed(ex_var, percent_var)
        event_trajectories = event_slice(t_mat, key, no_pcs, mode="single")
        temp_pairwise_distances = geodesic_distances(event_trajectories, mode="single")
        if is_first:
            pairwise_distances = temp_pairwise_distances
            is_first = False
        else:
            for pair, distance in temp_pairwise_distances.items():
                temp_distances = pairwise_distances[pair]
                temp_distances.append(temp_pairwise_distances[pair][0])
                pairwise_distances[pair] = temp_distances
    return pairwise_distances
