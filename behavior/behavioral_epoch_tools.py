import os
import numpy as np
from collections import defaultdict
from itertools import combinations
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random


def random_event_generator(start, stop, len_event, no_events):
    total_duration = stop - start
    possible_events = np.arange(int(total_duration / len_event))
    pot_events = np.random.choice(possible_events, size=(no_events), replace=False)
    pot_events = np.sort(pot_events)
    events = []
    for i in pot_events:
        event_start = start + (len_event * i)
        event_stop = event_start + (len_event)
        events.append(np.array([event_start, event_stop]))
    return np.array(events)


def threshold_bouts(start_stop_array, min_iti, min_bout):
    """
    thresholds behavior bouts by combining behavior bouts with interbout intervals of
    < min_iti and then removing remaining bouts of < min_bout

    Args (3 total):
        start_stop_array: numpy array of dim (# of bouts, 2)
        min_iti: float, min interbout interval in seconds
        min_bout: float, min bout length in seconds

    Returns (1):
        start_stop_array: numpy array (ndim=(n bouts, 2))
            of start&stop times
    """
    if isinstance(start_stop_array, list):
        start_stop_array = np.array(start_stop_array)
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


def find_overlapping_groups(eventA, eventB):
    all_events = np.concatenate([eventA, eventB])
    labels = np.concatenate([len(eventA) * ["A"], len(eventB) * ["B"]])

    # Step 2: Sort by start time
    sorted_indices = np.argsort(all_events[:, 0])
    sorted_events = all_events[sorted_indices]
    sorted_labels = labels[sorted_indices]

    # Step 3: Group overlapping intervals
    groups = []
    current_group = [(sorted_labels[0], tuple(sorted_events[0]))]
    current_max_stop = sorted_events[0][1]

    for i in range(1, len(sorted_events)):
        start, stop = sorted_events[i]
        label = sorted_labels[i]
        event_tuple = (label, (start, stop))

        if start <= current_max_stop:
            current_group.append(event_tuple)
            current_max_stop = max(current_max_stop, stop)
        else:
            groups.append(current_group)
            current_group = [event_tuple]
            current_max_stop = stop

    groups.append(current_group)  # Add the last group
    return groups


def check_overlap_threshold(group, overlap_threshold):
    time_points = prep_data(group)
    # Step 3: Sweep line
    overlap = 0
    current_active = 0
    prev_time = None

    for time, typ in time_points:
        if prev_time is not None and current_active >= 2:
            overlap += time - prev_time
        if typ == "start":
            current_active += 1
        else:
            current_active -= 1
        prev_time = time
    percent_overlap = overlap / (time_points[-1][0] - time_points[0][0])
    if percent_overlap >= overlap_threshold:
        return group
    else:
        return None


def prep_data(group):
    time_points = []
    for label, (start, end) in group:
        time_points.append((start, "start"))
        time_points.append((end, "end"))

    # Step 2: Sort by time (with 'start' before 'end' to handle boundaries correctly)
    time_points.sort(key=lambda x: (x[0], 0 if x[1] == "start" else 1))
    return time_points


def split_events(groups, event_dict, return_nonoverlap):
    event3 = []
    if return_nonoverlap:
        eventA = []
        eventB = []
        for group in groups:
            # Track events with unique IDs to handle multiple As or Bs
            time_points = []
            for idx, (label, (start, end)) in enumerate(group):
                event_id = (label, idx)
                time_points.append((start, "start", label, event_id))
                time_points.append((end, "end", label, event_id))

            # Sort time points
            time_points.sort(key=lambda x: (x[0], 0 if x[1] == "start" else 1))

            # Track active events
            active_events = set()
            active_labels = {"A": 0, "B": 0}
            prev_time = None

            for time, typ, label, event_id in time_points:
                if prev_time is not None and time > prev_time:
                    countA = active_labels["A"]
                    countB = active_labels["B"]
                    if countA > 0 and countB == 0:
                        eventA.append([prev_time, time])
                    elif countB > 0 and countA == 0:
                        eventB.append([prev_time, time])
                    elif countA > 0 and countB > 0:
                        event3.append([prev_time, time])
                    # If neither are active, we ignore

                # Update active sets
                if typ == "start":
                    active_events.add(event_id)
                    active_labels[label] += 1
                else:
                    active_events.discard(event_id)
                    active_labels[label] -= 1

                prev_time = time
        event_dict["eventA"] = np.array(eventA)
        event_dict["eventB"] = np.array(eventB)
    event_dict["event3"] = np.array(event3)
    return event_dict


def overlap_events(groups, event_dict):
    event3 = []
    for group in groups:
        time_points = prep_data(group)

        overlapping_chunks = []
        current_active = 0
        prev_time = None

        for time, typ in time_points:
            if prev_time is not None and current_active >= 2:
                overlapping_chunks.append([prev_time, time])
            if typ == "start":
                current_active += 1
            else:
                current_active -= 1
            prev_time = time

        event3.extend(overlapping_chunks)
    event_dict["event3"] = np.array(event3)
    return event_dict


def combine_events(groups, event_dict):
    event3 = []
    for group in groups:
        time_points = prep_data(group)
        event3.append([time_points[0][0], time_points[-1][0]])
    event_dict["event3"] = np.array(event3)
    return event_dict


def overlapping_events(eventA, eventB, overlap_threshold, mode, return_nonoverlap=False):
    event_dict = defaultdict(list)
    if mode == "duplicate":
        event_dict = duplicate_events(eventA, eventB, overlap_threshold, event_dict, return_nonoverlap)
    else:
        groups = find_overlapping_groups(eventA, eventB)
        good_groups = []

        included_events = set()
        for group in groups:
            if len(group) == 1:
                if return_nonoverlap:
                    if group[0][0] == "A":
                        event_dict["eventA"].append(group[0][1])
                    if group[0][0] == "B":
                        event_dict["eventB"].append(group[0][1])
            else:
                good_group = check_overlap_threshold(group, overlap_threshold)
                if good_group is not None:
                    good_groups.append(group)
                    if return_nonoverlap:
                        for event in good_group:
                            included_events.add(event)
                else:
                    if len(group) > 2:
                        event_pairs = []
                        for pair in combinations(group, 2):
                            event_pairs.append(pair)
                        for pair in event_pairs:
                            good_group = check_overlap_threshold(pair, overlap_threshold)
                            if good_group is not None:
                                good_groups.append(good_group)
                                if return_nonoverlap:
                                    for event in good_group:
                                        included_events.add(event)
        if len(good_groups) == 0:
            print("no overlap")
        if return_nonoverlap:
            for label, interval in zip(len(eventA) * ["A"] + len(eventB) * ["B"], list(eventA) + list(eventB)):
                event_tuple = (label, tuple(interval))
                if event_tuple not in included_events:
                    event_dict[f"event{label}"].append(tuple(interval))
        if mode == "combine":
            event_dict = combine_events(good_groups, event_dict)
        if mode == "split":
            event_dict = split_events(good_groups, event_dict, return_nonoverlap)
        if mode == "overlap":
            event_dict = overlap_events(good_groups, event_dict)
    return event_dict


def duplicate_events(eventA, eventB, overlap_threshold, event_dict, return_nonoverlap):
    """
    For each event in eventA, compute overlap with all eventB.
    Label as '1' if percent overlap >= threshold, else 'A'.
    Return list of tuples: (label, (start, stop))
    """

    for startA, stopA in eventA:
        durationA = stopA - startA
        total_overlap = 0

        for startB, stopB in eventB:
            # Calculate overlap between eventA and eventB intervals
            overlap_start = max(startA, startB)
            overlap_stop = min(stopA, stopB)
            overlap_duration = max(0, overlap_stop - overlap_start)
            total_overlap += overlap_duration

        percent_overlap = total_overlap / durationA if durationA > 0 else 0

        if percent_overlap >= overlap_threshold:
            event_dict["event1"].append([startA, stopA])  # overlapping label
        else:
            if return_nonoverlap:
                event_dict["eventA"].append([startA, stopA])
        # non-overlapping label

    for startB, stopB in eventB:
        durationB = stopB - startB
        total_overlap = 0

        for startA, stopA in eventA:
            # Calculate overlap between eventA and eventB intervals
            overlap_start = max(startB, startA)
            overlap_stop = min(stopB, stopA)
            overlap_duration = max(0, overlap_stop - overlap_start)
            total_overlap += overlap_duration

        percent_overlap = total_overlap / durationB if durationB > 0 else 0

        if percent_overlap >= overlap_threshold:
            event_dict["event2"].append([startB, stopB])  # overlapping label
        else:
            if return_nonoverlap:
                event_dict["eventB"].append([startB, stopB])
    event_dict = {event: np.array(v) for event, v in event_dict.items()}
    return event_dict


def first_eventA_after_eventB(eventA, eventB, overlap=False, delay=0):
    """
    For each event in eventB, return the first eventA that starts after (startB + delay).

    Parameters:
        eventA (array-like): Nx2 array of [start, stop] for eventA.
        eventB (array-like): Mx2 array of [start, stop] for eventB.
        overlap (bool): If True, include eventA that overlaps eventB onset.
        delay (float): Delay in seconds to apply to eventB onset before searching.

    Returns:
        np.ndarray: Array of matched eventA intervals.
    """
    eventA = np.array(eventA)
    eventB = np.array(eventB)
    matched_events = []

    for startB, _ in eventB:
        search_time = startB + delay
        # Mask for events that meet the condition
        if overlap:
            candidates = eventA[
                ((eventA[:, 0] <= search_time) & (eventA[:, 1] >= search_time)) | (eventA[:, 0] >= search_time)
            ]
        else:
            candidates = eventA[eventA[:, 0] >= search_time]

        if len(candidates) > 0:
            # Take the first one chronologically
            first_event = candidates[np.argmin(candidates[:, 0])]
            matched_events.append(first_event)
            matched_events = list(set(map(tuple, matched_events)))  # remove potential duplicates
            matched_events.sort()  # Optional: sort by start time

    return np.array(matched_events)


def plot_event_bars(eventA, event_dict, eventB=None, title="Event Plot"):
    """
    Plots original A and B at the top of the figure, followed by other event types.

    Parameters:
    - eventA: list of [start, stop] intervals
    - eventB: list of [start, stop] intervals
    - event_dict: dict of {label: list of [start, stop] intervals}
    - title: title for the plot
    """
    fig, ax = plt.subplots(figsize=(10, 0.6 * (2 + len(event_dict))))

    # Put original A and B at the beginning
    if eventB is not None:
        full_dict = {"original A": eventA, "original B": eventB}

    else:
        full_dict = {"original A": eventA}
    full_dict.update(event_dict)

    # Assign a unique color to each label
    color_map = {label: [random.random() for _ in range(3)] for label in full_dict}

    # Plot from top to bottom by reversing the order of plotting
    labels = list(full_dict.keys())
    n_labels = len(labels)

    yticks = []
    yticklabels = []

    for i, label in enumerate(labels):
        y = n_labels - 1 - i  # Reverse y-position so first item is at top
        for start, stop in full_dict[label]:
            ax.barh(y, stop - start, left=start, height=0.4, color=color_map[label], edgecolor="black")
        yticks.append(y)
        yticklabels.append(label)

    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel("Time")
    ax.set_title(title)
    ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    # Legend
    # legend_handles = [mpatches.Patch(color=color_map[k], label=k) for k in labels]
    # ax.legend(handles=legend_handles, loc='upper right')

    plt.tight_layout()
    plt.show()
