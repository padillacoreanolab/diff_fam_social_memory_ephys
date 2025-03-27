import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns
from bidict import bidict
from collections import defaultdict
from matplotlib import cm
from itertools import combinations
from lfp.lfp_analysis.LFP_collection import LFPCollection
from lfp.lfp_analysis.LFP_recording import LFPRecording

def all_set(collection):
    """
    double checks that all lfp objects in the collection have
    the attributes: subject & behavior_dict assigned to them and that
    each behavior_dict has the same keys.

    Prints statements telling user which recordings are missing subjects
    or behavior_dicts.
    Prints behavior_dict.keys() if they are not the same.
    Prints "All set to analyze" and calculates spiketrains
    and firing rates if all set.
    """
    is_first = True
    is_good = True
    missing_events = []
    missing_subject = []
    behavior_dicts_same = True
    for i in range(len(collection.lfp_recordings)):
        recording = collection.lfp_recordings[i]
        if not hasattr(recording, "behavior_dict"):
            missing_events.append(recording)
        else:
            if is_first:
                last_recording_events = recording.behavior_dict.keys()
                is_first = False
            else:
                if recording.behavior_dict.keys() != last_recording_events:
                    behavior_dicts_same = False
        if not hasattr(recording, "subject"):
            missing_subject.append(recording.name)
    if len(missing_events) > 0:
        print("These recordings are missing event dictionaries:")
        print(f"{missing_events}")
        is_good = False
    else:
        if not behavior_dicts_same:
            print("Your event dictionary keys are different across recordings.")
            print("Please double check them:")
            for recording in collection.lfp_recordings:
                print(recording.name, "keys:", recording.behavior_dict.keys())
    if len(missing_subject) > 0:
        print(f"These recordings are missing subjects: {missing_subject}")
        is_good = False
    if is_good:
        print("All set to analyze")


def _create_coherence_bidict(self):
    pairs_to_indices = bidict()
    region_to_index = self.brain_region_dict
    for i in range(len(region_to_index)):
        for j in range(i + 1, len(region_to_index)):
            regions = frozenset([list(region_to_index.keys())[i], list(region_to_index.keys())[j]])
            indices = (region_to_index[list(regions)[0]], region_to_index[list(regions)[1]])
            pairs_to_indices[regions] = indices
    return pairs_to_indices


def __get_events__(recording, event, mode, event_len, pre_window, post_window, average = True):
    """
    takes snippets of power, coherence, or causality for events
    optional pre-event and post-event windows (s) may be included
    all events can also be of equal length by extending
    snippet lengths to the longest event

    Args (6 total, 4 required):
        recording: LFP object instance, recording to get snippets
        event: str, event type of which ehpys snippets happen during
        whole_recording: numpy array, power, coherence, or granger causality
            for the whole recording
        event_len: optional, float, length (s) of events used by padding with
            post event time or trimming events all to event_len (s) long, if not
            defined, full event is used
        pre_window: int, default=0, seconds prior to start of event
        post_window: int, default=0, seconds after end of event

    Returns (1):
        event_averages: list, event specific measures of
            power, coherence, or casualities measures during an event including
            pre_window & post_windows, accounting for event_len and
            timebins; if mode is power, event_snippets has
            dimensions of [e, t, f, b] where e = no of events, b = no. of
            brain regions, t = no. of timebins, f = no. of frequencies
            if mode is causality or coherence then event snippets has the
            shape [e, t, f, b, b]
    """
    try:
        events = recording.behavior_dict[event]
    except KeyError:
        print(f"{event} not in event dictionary. Please check spelling")
    all_events = []
    pre_window = math.ceil(pre_window * 1000)
    post_window = math.ceil(post_window * 1000)
    freq_timebin = recording.timestep * 1000
    if event_len is not None:
        event_len_ms = event_len * 1000
    if mode == "power":
        whole_recording = recording.power
    if mode == "granger":
        whole_recording = recording.grangers
    if mode == "coherence":
        whole_recording = recording.coherence
    for i in range(events.shape[0]):
        if event_len is not None:
            pre_event = math.ceil((events[i][0] - pre_window) / freq_timebin)
            post_event = math.ceil((events[i][0] + post_window + event_len_ms) / freq_timebin)
        if event_len is None:
            pre_event = math.ceil((events[i][0] - pre_window) / freq_timebin)
            post_event = math.ceil((events[i][1] + post_window) / freq_timebin)
        if post_event < whole_recording.shape[0]:
            # whole_recording = [t, f, b]  for power
            # whole_recording = [t,f,b,b] for coherence + granger
            event_snippet = whole_recording[pre_event:post_event, ...]
            if average:
                event_snippet = np.nanmean(event_snippet, axis=0)
            all_events.append(event_snippet)
    return all_events


def plot_average_events(lfp_collection, event_averages, mode, regions=None, freq_range = None):
    if mode == "power":
        plot_power_averages(lfp_collection, event_averages, regions, freq_range)
    if mode == "coherence":
        plot_coherence_averages(lfp_collection, event_averages, regions, freq_range)
    if mode == "granger":
        plot_granger_averages(lfp_collection, event_averages, regions, freq_range)


def plot_power_averages(lfp_collection, event_averages, regions=None, freq_range=None):
    if regions is None:
        regions = lfp_collection.brain_region_dict.keys()
    if freq_range is None:
        freq_range = [1,101]
    for region in regions:
        plt.figure(figsize=(10, 5))
        for event, averages in event_averages.items():
            # averages = [trials, f, b]
            averages = event_averages[event]
            event_average = np.nanmean(averages, axis=0)
            # event_average = [f,b]; average across all trials
            # calculate sem for the trial average
            event_sem = stats.sem(averages, axis=0, nan_policy="omit")
            region_index = lfp_collection.brain_region_dict[region]
            # pick only the region of interest
            y = event_average[freq_range[0]:freq_range[1], region_index]
            y_sem = event_sem[freq_range[0]:freq_range[1], region_index]
            x = range(freq_range[0],freq_range[1])
            (line,) = plt.plot(x, y, label=event)
            plt.fill_between(x, y - y_sem, y + y_sem, alpha=0.2, color=line.get_color())
        ymin, ymax = plt.ylim()
        # plt.axvline(x=12, color="gray", linestyle="--", linewidth=0.5)
        # plt.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)
        # plt.fill_betweenx(y=np.linspace(ymin, ymax, 80), x1=4, x2=12, color="red", alpha=0.1)
        plt.ylim(ymin, ymax)
        plt.title(f"{region} power")
        plt.legend()
        plt.show()


def plot_coherence_averages(lfp_collection, event_averages, regions=None, freq_range=None):
    if regions is not None:
        pairs_indices = []
        for region in regions:
            try:
                pairs_index = lfp_collection.coherence_pairs_dict[frozenset({region[0], region[1]})]
            except KeyError:
                pairs_index = lfp_collection.coherence_pairs_dict[frozenset({region[0], region[1]})]
            pairs_indices.append(pairs_index)
    if regions is None:
        regions = list(lfp_collection.coherence_pairs_dict.values())
    if freq_range is None:
        freq_range = [1,101]
    for i in range(len(regions)):
        for event, averages in event_averages.items():
            # averages = [trials, f, b, b]
            first_region, second_region = list(regions[i])
            first_region_name = lfp_collection.brain_region_dict.inverse[first_region]
            second_region_name = lfp_collection.brain_region_dict.inverse[second_region]
            averages = event_averages[event]
            event_average = np.nanmean(averages, axis=0)
            # event_average = [f, b, b]; average across all trials
            # calculate sem for the trial average
            event_sem = stats.sem(averages, axis=0, nan_policy="omit")
            # pick only the region of interest
            y_sem = event_sem[freq_range[0]:freq_range[1], first_region, second_region]
            y = event_average[freq_range[0]:freq_range[1], first_region, second_region]
            x = lfp_collection.frequencies[freq_range[0]:freq_range[1]]
            (line,) = plt.plot(x, y, label=event)
            plt.fill_between(x, y - y_sem, y + y_sem, color=line.get_color(), alpha=0.2)
        ymin, ymax = plt.ylim()
        plt.axvline(x=12, color="gray", linestyle="--", linewidth=0.5)
        plt.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)
        plt.fill_betweenx(y=np.linspace(ymin, ymax, 100), x1=4, x2=12, color="red", alpha=0.1)
        plt.ylim(ymin, ymax)
        plt.title(f"{first_region_name} & {second_region_name} coherence")
        plt.legend()
        plt.show()


def plot_granger_averages(lfp_collection, event_averages, regions=None, freq_range=None):
    if regions is not None:
        pair_indices = []
        for region in regions:
            first_index = lfp_collection.brain_region_dict[region[0]]
            second_index = lfp_collection.brain_region_dict[region[1]]
            pair_indices.append([first_index, second_index])
    if regions is None:
        pair_indices = list(permutations(range(len(lfp_collection.brain_regions)), 2))
        regions = []
        for pair in pair_indices:
            regions.append([lfp_collection.brain_regions[pair[0]], lfp_collection.brain_regions[pair[1]]])
    if freq_range is None:
        freq_range = [1,101]
    for i in range(len(pair_indices)):
        for event, averages in event_averages.items():
            # averages = [trials, b, b, f]
            region = regions[i]
            averages = event_averages[event]
            event_average = np.nanmean(averages, axis=0)
            # event_average = [b,b,f]; average across all trials
            # calculate sem for the trial average
            event_sem = stats.sem(averages, axis=0, nan_policy="omit")
            # pick only the region of interest
            y_sem = event_sem[freq_range[0]:freq_range[1], pair_indices[i][0], pair_indices[i][1]]
            y = event_average[freq_range[0]:freq_range[1], pair_indices[i][0], pair_indices[i][1]]
            x = lfp_collection.frequencies[freq_range[0]:freq_range[1]]
            (line,) = plt.plot(x, y, label=event)
            plt.fill_between(x, y - y_sem, y + y_sem, color=line.get_color(), alpha=0.2)
        ymin, ymax = plt.ylim()
        plt.axvline(x=12, color="gray", linestyle="--", linewidth=0.5)
        plt.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)
        plt.fill_betweenx(y=np.linspace(ymin, ymax, 80), x1=4, x2=12, color="red", alpha=0.1)
        plt.ylim(ymin, ymax)
        plt.title(f"Granger causality: {region[0]} to {region[1]}")
        plt.legend()
        plt.show()



def plot_granger_heatmap(lfp_collection, events, freq, baseline=None, event_len=None):
    event_grangers = average_events(lfp_collection,events, mode="granger", baseline=baseline, event_len=event_len, plot=False)
    n_events = len(events)
    n_cols = min(3, n_events)  # Max 3 columns
    n_rows = (n_events + n_cols - 1) // n_cols  # Ceiling division
    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_events == 1:
        axes = np.array([axes])  # Make single axis iterable
    axes = axes.flatten()  # Flatten for easy iteration
    # Get brain regions once since they're the same for all plots
    brain_regions = np.empty(len(lfp_collection.brain_region_dict.keys()), dtype="<U10")
    for i in range(len(lfp_collection.brain_region_dict.keys())):
        brain_regions[i] = lfp_collection.brain_region_dict.inverse[i]
    for idx, (event, ax) in enumerate(zip(events, axes)):
        event_granger = event_grangers[event]
        avg_granger = np.nanmean(event_granger, axis=0)
        freq_granger = avg_granger[freq[0] : freq[1], :, :]
        avg_freq = np.nanmean(freq_granger, axis=0)
        sns.heatmap(avg_freq, xticklabels=brain_regions, yticklabels=brain_regions, annot=True, cmap="viridis", ax=ax)
        ax.set_title(f"{event} Granger Causality\n{freq[0]}Hz to {freq[1]}Hz")
        # Rotate labels for better readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
        ax.set_ylabel("From")
        ax.set_xlabel("To")
    # Remove any empty subplots
    for idx in range(len(events), len(axes)):
        fig.delaxes(axes[idx])
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    
# def plot_spectogram(lfp_collection, events, mode, event_length, baseline=None, pre_window = 0, post_window = 0):
#     event_averages_dict = {}
#     if isinstance(lfp_collection , LFPCollection):
#         recordings = lfp_collection.lfp_recordings
#     if isinstance(lfp_collection , LFPRecording):
#         recordings = [lfp_collection]
#     for event in events:
#         recording_averages = []
#         for recording in recordings:
#             #events = [trials, time, freq, regions]
#             events = __get_events__(recording, event, mode, event_len, pre_window, post_window, average = False)
#             if baseline is not None:
#                 #TODO: edit this to work with not averages 
#                 adj_averages = __baseline_diff__(
#                     recording, event_averages, baseline, mode, event_len, pre_window=0, post_window=0
#                 )
#                 recording_averages = recording_averages + adj_averages
#             else:
#                 recording_averages = recording_averages + event_averages
            

def average_events(
    lfp_collection, events, mode, baseline=None, event_len=None, pre_window=0, post_window=0, plot=False, regions = None, freq_range = None
):
    """
    Calculates average event measurement (power, coherence, or granger) per recording then
    calculates global averages across all recordings from recording averages (to account for
    differences in event numbers per recording)
    """
    event_averages_dict = {}
    if isinstance(lfp_collection , LFPCollection):
        recordings = lfp_collection.lfp_recordings
    if isinstance(lfp_collection , LFPRecording):
        recordings = [lfp_collection]
    for event in events:
        recording_averages = []
        for recording in recordings:
            event_averages = __get_events__(recording, event, mode, event_len, pre_window, post_window, average = True)
            if baseline is not None:
                adj_averages = __baseline_diff__(
                    recording, event_averages, baseline, mode, event_len, pre_window=0, post_window=0, average = True
                )
                recording_averages = recording_averages + adj_averages
            else:
                recording_averages = recording_averages + event_averages
        # recording_averages = [trials, b, f] or [trials, b, b, f]
        event_averages_dict[event] = recording_averages
    if plot:
        plot_average_events(lfp_collection, event_averages_dict, mode, regions, freq_range)
    return event_averages_dict


def __baseline_diff__(recording, event_averages, baseline, mode, event_len, pre_window, post_window, average):
    baseline_averages = __get_events__(recording, baseline, mode, event_len, pre_window, post_window, average = average)
    #average = trial, freq, regions
    #not average = trial, time, freq, regions
    baseline_recording = np.nanmean(np.nanmean(np.array(baseline_averages), axis=0), axis = 0)
    adj_averages = []
    for i in range(len(event_averages)):
        adj_average = ((event_averages[i] - baseline_recording) / (baseline_recording + 0.00001)) * 100
        adj_averages.append(adj_average)
    return adj_averages


def __get_event_snippets__(recording, event, mode, event_len, pre_window, post_window):
    """
    takes snippets of power, coherence, or causality for events
    optional pre-event and post-event windows (s) may be included
    all events can also be of equal length by extending
    snippet lengths to the longest event

    Args (6 total, 4 required):
        recording: LFP object instance, recording to get snippets
        event: str, event type of which ehpys snippets happen during
        whole_recording: numpy array, power, coherence, or granger causality
            for the whole recording
        event_len: optional, float, length (s) of events used by padding with
            post event time or trimming events all to event_len (s) long, if not
            defined, full event is used
        pre_window: int, default=0, seconds prior to start of event
        post_window: int, default=0, seconds after end of event

    Returns (1):
        event_averages: list, event specific measures of
            power, coherence, or casualities measures during an event including
            pre_window & post_windows, accounting for event_len and
            timebins; if mode is power of coherence, event_snippets has
            dimensions of [e, b, t, f] where e = no of events, b = no. of
            brain regions, t = no. of timebins, f = no. of frequencies
            if mode is causality then event snippets has the shape [e, b, b, t, f]
    """
    try:
        events = recording.behavior_dict[event]
    except KeyError:
        print(f"{event} not in event dictionary. Please check spelling")
    event_snippets = []
    pre_window = round(pre_window * 1000)
    post_window = round(post_window * 1000)
    freq_timebin = recording.timewindow * 1000
    if event_len is not None:
        event_len_ms = event_len * 1000
        event_dur = int(event_len_ms + pre_window + post_window) / freq_timebin
    if mode == "power":
        whole_recording = recording.power
    if mode == "granger":
        whole_recording = recording.grangers
    if mode == "coherence":
        whole_recording = recording.coherence
    for i in range(events.shape[0]):
        if event_len is not None:
            pre_event = int((events[i][0] - pre_window) / freq_timebin)
            post_event = int(pre_event + event_dur)
        if event_len is None:
            pre_event = math.ceil((events[i][0] - pre_window) / freq_timebin)
            post_event = math.ceil((events[i][1] + post_window) / freq_timebin)
        if mode != "granger":  # power is [b, t, f]; coherence is [bps, t, f]
            if post_event < whole_recording.shape[1]:
                event_snippet = whole_recording[:, pre_event:post_event, :]
                event_snippets.append(event_snippet)
        if mode == "granger":  # granger is (b, b, t, f)
            if post_event < whole_recording.shape[2]:
                event_snippet = whole_recording[:, :, pre_event:post_event, :]
                event_snippets.append(event_snippet)
    return event_snippets


def band_calcs(values):
    agent_band_dict = {}
    for agent, calculations in values.items():
        calculations = np.array(calculations)
        # calculations = [trials, frequencies, brain regions]
        delta = np.nanmean(calculations[:, 0:4, ...], axis=1)
        theta = np.nanmean(calculations[:, 4:13, ...], axis=1)

        beta = np.nanmean(calculations[:, 13:31, ...], axis=1)

        low_gamma = np.nanmean(calculations[:, 31:71, ...], axis=1)

        high_gamma = np.nanmean(calculations[:, 71:100, ...], axis=1)

        agent_band_dict[agent] = {
            "Delta": delta,
            "Theta": theta,
            "Beta": beta,
            "Low gamma": low_gamma,
            "High gamma": high_gamma,
        }

    band_agent_dict = defaultdict(dict)
    for agent, bands in agent_band_dict.items():
        for band, values in bands.items():
            band_agent_dict[band][agent] = values

    return [agent_band_dict, band_agent_dict]


def average_power_bar(lfp_collection, events, baseline=None):
    powers = average_events(lfp_collection, events=events, mode="power", baseline=baseline, plot=False)
    [unflipped, flipped] = band_calcs(powers)
    brain_regions = np.empty(len(lfp_collection.brain_region_dict.keys()), dtype="<U10")
    for i in range(len(lfp_collection.brain_region_dict.keys())):
        brain_regions[i] = lfp_collection.brain_region_dict.inverse[i]
    avg_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}
    sem_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}
    for key in flipped.keys():
        for i, subset in enumerate(brain_regions):
            for event in events:
                avg_values[key][subset][event] = np.nanmean(flipped[key][event][:, i])
                sem_values[key][subset][event] = stats.sem(flipped[key][event][:, i], nan_policy="omit")

    # Adjust bar width and spacing based on number of events
    total_width = 0.8  # Total width available for each group of bars
    bar_width = total_width / len(events)  # Width of each bar
    col = cm.rainbow(np.linspace(0, 1, len(events)))

    # Spacing between groups of bars
    group_spacing = 1  # Increased for better separation between brain regions

    sorted_avg_values = {key: {subset: avg_values[key][subset] for subset in brain_regions} for key in flipped.keys()}
    sorted_sem_values = {key: {subset: sem_values[key][subset] for subset in brain_regions} for key in flipped.keys()}

    # Create a separate plot for each key
    for key in flipped.keys():
        plt.figure(figsize=(25, 10))
        x = np.arange(len(brain_regions)) * group_spacing  # x-axis positions for subsets

        for i, subset in enumerate(brain_regions):
            for k, event in enumerate(events):
                # Center the group of bars and space them evenly
                center = x[i]
                offset = (k - (len(events) - 1) / 2) * bar_width
                position = center + offset

                plt.bar(
                    position,
                    sorted_avg_values[key][subset][event],
                    width=bar_width,
                    yerr=sorted_sem_values[key][subset][event],
                    capsize=5,
                    linewidth=2,
                    error_kw={"elinewidth": 2, "capthick": 2},
                    color=col[k],
                    label=event if i == 0 else "",
                )

        plt.yticks(fontsize=20)
        plt.xticks(x, brain_regions, fontsize=24, rotation=45)
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.8)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_linewidth(2)
        plt.gca().spines["left"].set_linewidth(2)
        plt.title(f"Average Power for {key}", fontsize=40)
        plt.legend(fontsize=26, frameon=False)
        plt.subplots_adjust(hspace=0.5)
        plt.show()


def plot_coherence_bar(lfp_collection, events, baseline=None):
    region_dict = lfp_collection.brain_region_dict
    brain_regions = list(combinations(list((region_dict.keys())), 2))  # Example subset names

    coherences = average_events(lfp_collection, events=events, mode="coherence", baseline=baseline, plot=False)
    [unflipped, flipped] = band_calcs(coherences)
    avg_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}
    sem_values = {key: {subset: {event: [] for event in events} for subset in brain_regions} for key in flipped.keys()}

    for key in flipped.keys():
        for i, subset in enumerate(brain_regions):
            pair_index_1 = region_dict[brain_regions[i][0]]
            pair_index_2 = region_dict[brain_regions[i][1]]
            for event in events:
                avg_values[key][subset][event] = np.nanmean(flipped[key][event][:, pair_index_1, pair_index_2])
                sem_values[key][subset][event] = stats.sem(
                    flipped[key][event][:, pair_index_1, pair_index_2], nan_policy="omit"
                )

    # Width of each bar
    col = cm.rainbow(np.linspace(0, 1, len(events)))
    spacing = 0
    # Create a separate plot for each key
    # Spacing between different subsets
    total_width = 0.8  # Total width available for each group of bars
    bar_width = total_width / len(events)
    group_spacing = 1
    # Create a separate plot for each key
    for key in flipped.keys():
        plt.figure(figsize=(25, 10))
        x = np.arange(len(brain_regions)) * group_spacing  # x-axis positions for subsets

        for i, subset in enumerate(brain_regions):
            for k, event in enumerate(events):
                positions = x[i] + (k - 1) * (bar_width + spacing)  # Adjust positions for each event
                plt.bar(
                    positions,
                    avg_values[key][subset][event],
                    width=bar_width,
                    yerr=sem_values[key][subset][event],
                    capsize=5,
                    linewidth=2,
                    error_kw={"elinewidth": 2, "capthick": 2},
                    color=col[k],
                    label=event if i == 0 else "",
                )

        plt.yticks(fontsize=16)
        plt.xticks(x, brain_regions, fontsize=18, rotation=45)
        plt.axhline(y=0, color="black", linestyle="--", alpha=0.8)
        plt.ylabel("Average Coherence", fontsize=20)
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().spines["bottom"].set_linewidth(2)  # X-axis
        plt.gca().spines["left"].set_linewidth(2)
        plt.title(f"Average Coherence for {key}", fontsize=26, font="Arial")
        plt.legend(fontsize=16, frameon=False)
        plt.subplots_adjust(hspace=0.5)
        plt.show()
