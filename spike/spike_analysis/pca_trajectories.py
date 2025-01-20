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


def pca_matrix(spike_collection, event_length, pre_window, post_window, events, mode, min_events=None):
    event_keys = []
    recording_keys = []
    pca_master_matrix = None
    num_points = int((event_length + pre_window + post_window) * 1000 / spike_collection.timebin)
    if events is None:
        events = spike_collection.collection[0].event_dict.keys()
    for recording in spike_collection.collection:
        pca_matrix = None
        for event in events:
            firing_rates = recording.__event_firing_rates__(event, event_length, pre_window, post_window)
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
    return pca_dict(pca_master_matrix, recording_keys, event_keys)


def avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window=0, events=None):
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
    return pca_matrix(spike_collection, event_length, pre_window, post_window, events, mode="average", min_events=None)


def trial_trajectory_matrix(spike_collection, event_length, pre_window, post_window=0, events=None):
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
    min_events = event_numbers(spike_collection, events)
    return pca_matrix(
        spike_collection, event_length, pre_window, post_window, events, mode="trial", min_events=min_events
    )


def event_numbers(spike_collection, events):
    mins = {}
    if events is None:
        events = list(spike_collection.collection[0].event_dict.keys())
    for event in events:
        totals = []
        for recording in spike_collection.collection:
            totals.append((recording.event_dict[event]).shape[0])
        mins[event] = min(totals)
    print(mins)
    return mins


def pca_dict(matrix, recording_keys, event_keys):
    matrix_df = pd.DataFrame(data=matrix, columns=recording_keys, index=event_keys)
    key = np.array(matrix_df.index.to_list())
    if matrix.shape[0] < matrix.shape[1]:
        print("Warning: you have more features (neurons) than samples (time bins)")
        print("Consider choosing a smaller time window for analysis")
        pca_dict = {
            "raw data": matrix_df,
            "transformed data": None,
            "labels": key,
            "coefficients": None,
            "explained variance": None,
        }
    else:
        pca = PCA()
        pca.fit(matrix_df)
        pca_dict = {
            "raw data": matrix_df,
            "transformed data": pca.transform(matrix_df),
            "labels": key,
            "coefficients": pca.components_,
            "explained variance": pca.explained_variance_ratio_,
        }
    return pca_dict


def avg_trajectories_pca(
    spike_collection, event_length, pre_window, post_window=0, events=None, plot=True, d=2, azim=30, elev=20
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
    pc_dict = avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events)
    if plot:
        if d == 2:
            avg_trajectory_EDA_plot(
                spike_collection, pc_dict["transformed data"], pc_dict["labels"], event_length, pre_window, post_window
            )
        if d == 3:
            avg_trajectory_EDA_plot_3D(
                spike_collection,
                pc_dict["transformed data"],
                pc_dict["labels"],
                event_length,
                pre_window,
                post_window,
                azim,
                elev,
            )
    return pc_dict


def trial_trajectories_pca(
    spike_collection, event_length, pre_window=0, post_window=0, events=None, plot=True, d=2, azim=30, elev=20
):
    pc_dict = trial_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events)
    min_events = event_numbers(spike_collection, events)
    if plot:
        if d == 2:
            trial_trajectory_EDA_plot(
                spike_collection,
                pc_dict["transformed data"],
                pc_dict["labels"],
                event_length,
                pre_window,
                post_window,
                min_events,
            )
        if d == 3:
            trial_trajectory_EDA_3D_plot(
                spike_collection,
                pc_dict["transformed data"],
                pc_dict["labels"],
                event_length,
                pre_window,
                post_window,
                min_events,
                azim,
                elev,
            )
    return pc_dict


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
    # conv_factor = 1000 / spike_collection.timebin
    # timebins_per_trial = int((event_length + pre_window + post_window) * conv_factor)
    # event_end = int((event_length + pre_window) * conv_factor)
    # pre_window = pre_window * conv_factor
    # post_window = post_window * conv_factor
    # alpha = 0.5
    # marker_size = 5
    # highlight_markers = True
    # # Get unique events and assign colors
    # unique_events = list(set(PCA_key))
    # colors = plt.cm.tab10(np.linspace(0, 1, len(unique_events)))
    # color_dict = dict(zip(unique_events, colors))

    # # Create 3D plot
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection="3d")

    # # Plot each trial
    # for i in range(0, len(PCA_key), timebins_per_trial):
    #     event_label = PCA_key[i]
    #     color = color_dict[event_label]

    #     # Calculate marker positions for this trial
    #     onset = i if pre_window == 0 else int(i + pre_window - 1)
    #     end = int(i + event_end - 1)
    #     post = int(i + timebins_per_trial - 1)

    #     # Plot trajectory
    #     ax.plot(
    #         pca_matrix[i : i + timebins_per_trial, 0],
    #         pca_matrix[i : i + timebins_per_trial, 1],
    #         pca_matrix[i : i + timebins_per_trial, 2],
    #         color=color,
    #         alpha=alpha,
    #         linewidth=0.8,
    #     )

    #     ax.scatter(
    #         pca_matrix[i : i + timebins_per_trial, 0],
    #         pca_matrix[i : i + timebins_per_trial, 1],
    #         pca_matrix[i : i + timebins_per_trial, 2],
    #         s=marker_size,
    #         color=color,
    #         alpha=alpha,
    #     )

    #     # Add event markers if requested
    #     if highlight_markers:
    #         marker_kwargs = dict(s=30, alpha=1, edgecolors=color, facecolors="none")

    #         # Start marker
    #         if pre_window != 0:
    #             ax.scatter(pca_matrix[i, 0], pca_matrix[i, 1], pca_matrix[i, 2], marker="s", **marker_kwargs)

    #         # Event onset marker
    #         ax.scatter(pca_matrix[onset, 0], pca_matrix[onset, 1], pca_matrix[onset, 2], marker="^", **marker_kwargs)

    #         # Event end marker
    #         ax.scatter(pca_matrix[end, 0], pca_matrix[end, 1], pca_matrix[end, 2], marker="o", **marker_kwargs)

    #         # Post-event marker if applicable
    #         if post_window != 0:
    #             ax.scatter(pca_matrix[post, 0], pca_matrix[post, 1], pca_matrix[post, 2], marker="D", **marker_kwargs)

    # # Add legend with one entry per event type
    # handles = [
    #     plt.Line2D([0], [0], color=color_dict[event], label=event, alpha=0.8, marker="o", markersize=5)
    #     for event in unique_events
    # ]
    # ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1, 1))

    # # Set labels and title
    # ax.set_xlabel("PC1")
    # ax.set_ylabel("PC2")
    # ax.set_zlabel("PC3")

    # post_win_text = ""
    # pre_win_text = ""
    # if post_window != 0:
    #     post_win_text = ", Post = ◇"
    # if pre_window != 0:
    #     pre_win_text = "Pre-event = □, "
    # title = pre_win_text + "Onset = △, End = ○" + post_win_text
    # plt.title(title)

    # Set viewing angle
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


def LOO_PCA(spike_collection, event_length, pre_window, percent_var, post_window=0, events=None):
    pc_dict = avg_trajectory_matrix(spike_collection, event_length, pre_window, post_window, events)
    full_pca_matrix = pc_dict["raw data"]
    key = pc_dict["labels"]
    coefficients = pc_dict["coefficients"]
    explained_variance_ratios = pc_dict["explained variance"]
    transformed_subsets = []
    i = 0
    recording_indices = get_indices(full_pca_matrix.columns.to_list())
    for i in range(len(recording_indices)):
        start = recording_indices[i][0]
        stop = recording_indices[i][1]
        subset_df = full_pca_matrix.drop(full_pca_matrix.columns[start:stop], axis=1)
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
