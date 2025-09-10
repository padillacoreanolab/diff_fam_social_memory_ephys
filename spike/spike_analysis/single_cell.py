import numpy as np
from statistics import StatisticsError
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.stats import sem, ranksums, fisher_exact, wilcoxon
import spike.spike_analysis.normalization as normalization


def pre_event_window(event, baseline_window, offset):
    """
    creates an event like object np.array[start(ms), stop(ms)] for
    baseline_window amount of time prior to an event

    Args (2 total):
        event: np.array[start(ms), stop(ms)]
        baseline_window: int, seconds prior to an event

    Returns (1):
        preevent: np.array, [start(ms),stop(ms)] baseline_window(s)
            before event
    """
    preevent = [event[0] - (baseline_window * 1000) - 1, event[0] + (offset * 1000) - 1]
    return np.array(preevent)


def event_avgs(event1_firing_rates, event2_firing_rates):
    unit_averages = {}
    bad_units = []
    for unit in event1_firing_rates.keys():
        try:
            event1_averages = [np.nanmean(event) for event in event1_firing_rates[unit]]
            event2_averages = [np.nanmean(event) for event in event2_firing_rates[unit]]
            unit_averages[unit] = [np.array(event1_averages), np.array(event2_averages)]
        except StatisticsError:
            bad_units.append(unit)
    return unit_averages, bad_units


def signed_rank(unit_averages):
    """
    unit_averages: dict
    keys: units (str)
    values: list of lists of averages per event for event 1 and event 2
        [[event 1 averages], [event 2 averages]]
    """
    for unit in unit_averages.keys():
        event1_averages, event2_averages = unit_averages[unit]
        min_length = min(len(event1_averages), len(event2_averages))
        event2_averages = event2_averages[:min_length]
        event1_averages = event1_averages[:min_length]
        unit_averages[unit] = [event1_averages, event2_averages]
    wilcoxon_stats = {}
    for unit in unit_averages.keys():
        if (not np.isnan(np.array(unit_averages[unit][0])).any()) or (
            not np.isnan(np.array(unit_averages[unit][1])).any()
        ):  # Check if data is valid before running Wilcoxon
            unit_averages_wil_array = np.array(unit_averages[unit][0]) - np.array(unit_averages[unit][1])
            # check what the dimensionality of this is
            unit_averages_wil_array_no_zeros = unit_averages_wil_array[unit_averages_wil_array != 0]
            results = wilcoxon(unit_averages_wil_array_no_zeros)
            wilcoxon_stats[unit] = {"Wilcoxon Stat": results.statistic, "p value": results.pvalue}
        else:
            wilcoxon_stats[unit] = {"Wilcoxon Stat": np.nan, "p value": np.nan}
    wilcox_df = dict_to_df(wilcoxon_stats)
    return wilcox_df


def rank_sum(unit_averages):
    wilcoxon_stats = {}
    for unit in unit_averages.keys():
        if (not np.isnan(np.array(unit_averages[unit][0])).any()) or (
            not np.isnan(np.array(unit_averages[unit][1])).any()
        ):
            results = ranksums(unit_averages[unit][0], unit_averages[unit][1])
            wilcoxon_stats[unit] = {"Wilcoxon Stat": results.statistic, "p value": results.pvalue}
        else:
            wilcoxon_stats[unit] = {"Wilcoxon Stat": np.nan, "p value": np.nan}
    wilcox_df = dict_to_df(wilcoxon_stats)
    return wilcox_df


def w_assessment(p_value, w):
    try:
        if p_value < 0.05:
            if w > 0:
                return "increases"
            else:
                return "decreases"
        else:
            return "not significant"
    except TypeError:
        return "NaN"


def dict_to_df(wilcox_dict):
    wilcoxon_df = pd.DataFrame.from_dict(wilcox_dict, orient="index")
    wilcoxon_df.columns = ["Wilcoxon Stat", "p value"]
    wilcoxon_df["event1 vs event2"] = wilcoxon_df.apply(
        lambda row: w_assessment(row["p value"], row["Wilcoxon Stat"]), axis=1
    )
    return wilcoxon_df


def wilcox_check(recording, unit_averages, bad_units):
    if len(bad_units) > 0:
        wilcox_warning(recording, bad_units)
    # for unit in unit_averages.keys():
    #     print(unit_averages[unit][0].shape)
    #     if unit_averages[unit][0] == unit_averages[unit][1]:
    #         print(f"Wilcoxon can't be done on {recording.name} {unit}, because baseline = event")
    #         unit_averages[unit] = [np.nan, np.nan]
    return unit_averages


def wilcox_warning(recording, bad_units):
    total_units = len(recording.freq_dict.keys())
    tot_badunits = len(bad_units)
    g_units = total_units = tot_badunits
    print(f"{recording.name} statistics done on {g_units} out of {total_units}")
    for i in range(len(bad_units)):
        bad_unit = bad_units[i]
        total_spikes = len(recording.unit_timestamps[bad_unit])
        print(f"unit {bad_unit} has too few spikes with {total_spikes} total spikes")


def wilcoxon_rec(recording, event, event_length, baseline_window, offset, exclude_offset):
    preevent_baselines = np.array(
        [pre_event_window(event, baseline_window, offset) for event in recording.event_dict[event]]
    )
    if len(recording.event_dict[event]) < 6:
        print(f"Wilcoxon can't be done on {recording.name} {event}, because <6 samples")
        return None
    tot_event = baseline_window + offset
    unit_baseline_firing_rates = recording.unit_event_firing_rates(preevent_baselines, tot_event)
    if exclude_offset:
        unit_event_firing_rates = recording.unit_event_firing_rates(event, event_length)
    else:
        unit_event_firing_rates = recording.unit_event_firing_rates(event, event_length, -(offset))
    unit_averages, bad_units = event_avgs(unit_event_firing_rates, unit_baseline_firing_rates)
    unit_averages = wilcox_check(recording, unit_averages, bad_units)
    wilcox_df = signed_rank(unit_averages)
    return wilcox_df


def wilcoxon_collection(
    spike_collection, event, event_length, baseline_window, offset=0, exclude_offset=False, plot=True
):
    """
    Runs a wilcoxon signed rank test on all good units from all recordings on the
    given event's firing rate versus the given baseline window immediately prior to event onset.

    Creates a dataframe with rows for each unit and columns representing Wilcoxon stats, p values, orginal unit ids,
    recording, subject and the event + baseline window.

    Args(4 total, 3 required):
        event: str, event firing rates for stats to be run on
        event_length: float, length (s) of events used by padding or trimming events all to event_length (s)
        baseline_window: int, default=0, seconds prior to start of event
        offset: int, adjusts end of baseline by offset(s) from onset of behavior i.e. offset=2 adds first 2s of
            event firing rates onto baseline, offest=-2 removes firing rate data from baseline averages
        exclude_offset: Boolean, default=False, if true excludes time prior to onset and before offset in event avgs,
            if false, time between onset and offset are included in event averages
            time between onset and offset are included in event averages
        plot: Boolean, default=True, if True, plots, if False, does not plot.

    Returns(1):
        master_df: df, rows for each unit and columns representing
            Wilcoxon stats, p values, orginal unit ids, recording
    """
    is_first = True
    for recording in spike_collection.recordings:
        recording_df = wilcoxon_rec(recording, event, event_length, baseline_window, offset, exclude_offset)
        if recording_df is not None:
            recording_df = recording_df.reset_index().rename(columns={"index": "original unit id"})
            recording_df["Recording"] = recording.name
            recording_df["Subject"] = recording.subject
            recording_df["Event"] = f"{event_length}s {event} vs {baseline_window}s baseline"
            if is_first:
                master_df = recording_df
                is_first = False
            else:
                master_df = pd.concat([master_df, recording_df], axis=0).reset_index(drop=True)
    if plot:
        baseline_v_event_plot(spike_collection, master_df, event, event_length, baseline_window, offset)
    return master_df[
        ["Event", "original unit id", "Wilcoxon Stat", "p value", "Recording", "Subject", "event1 vs event2"]
    ]


def wilcoxon_event1v2_rec(recording, event1, event2, event_length):
    """
    calculates wilcoxon signed-rank test for average firing rates between
    two events for a given recording. the resulting dataframe of wilcoxon stats
    and p values for every unit is added to a dictionary of dataframes for that
    recording.

    Key for this dictionary item is '{event1 } vs {event2} ({event_length}s)'
    and the value is the dataframe. Option to save for export.

    Args (5 total, 4 required):
        recording: EphysRecording instance
        event1: str, first event type firing rates for stats to be run on
        event2: str, second event type firing rates for stats to be run on
        event_length: float, length (s) of events used by padding with post event time
            or trimming events all to event_length (s) long used in stat

    Return (1):
        wilcoxon_df: pandas dataframe, columns are unit ids,
        row[0] are wilcoxon statistics and row[1] are p values

    """
    unit_event1_firing_rates = recording.unit_event_firing_rates(event1, event_length)
    unit_event2_firing_rates = recording.unit_event_firing_rates(event2, event_length)
    if (len(recording.event_dict[event1]) < 6) | (len(recording.event_dict[event2]) < 6):
        print(f"Wilcoxon can't be done on {recording.name} because <6 samples for either {event1} or {event2}")
        return None
    unit_averages, bad_units = event_avgs(unit_event1_firing_rates, unit_event2_firing_rates)
    unit_averages = wilcox_check(recording, unit_averages, bad_units)
    wilcoxon_df = rank_sum(unit_averages)
    return wilcoxon_df


def wilcoxon_event1v2_collection(spike_collection, event1, event2, event_length, pre_window, plot=True):
    """
    Runs a wilcoxon signed rank test on all good units of
    all recordings in the collection on the
    given event's firing rate versus another given event's firing rate.

    Creates a dataframe with rows for each unit and columns representing
    Wilcoxon stats, p values, orginal unit ids, recording,
    subject and the events given.  Dataframe is saved in the collections
    wilcox_dfs dictionary, key is '{event1} vs {event2}'

    Args(4 total, 3 required):
        event1: str, first event type firing rates for stats to be run on
        event2: str, second event type firing rates for stats to be run on
        event_length: float, length (s) of events used by padding with post event time
            or trimming events all to event_length (s) long used in stat
        pre_window: time prior to event onset to be included in plot
        plot: Boolean, default=True, if True, plots, if false, does not plot.
        save: Boolean, default=False, if True, saves results to wilcox_dfs attribute
              of the collection for export

    Returns (1):
        master_df: df, rows for each unit and columns representing
        Wilcoxon stats, p values, orginal unit ids, recording,
        subject and the events given
    """
    is_first = True
    for recording in spike_collection.recordings:
        recording_df = wilcoxon_event1v2_rec(recording, event1, event2, event_length)
        if recording_df is not None:
            recording_df = recording_df.reset_index().rename(columns={"index": "original unit id"})
            recording_df["Recording"] = recording.name
            recording_df["Subject"] = recording.subject
            recording_df["Event"] = f"{event1 } vs {event2} ({event_length}s)"
            if is_first:
                master_df = recording_df
                is_first = False
            else:
                master_df = pd.concat([master_df, recording_df], axis=0).reset_index(drop=True)
    if plot:
        event1v2_plot(spike_collection, master_df, event1, event2, event_length, pre_window)
    return master_df[
        ["Event", "original unit id", "Wilcoxon Stat", "p value", "Recording", "Subject", "event1 vs event2"]
    ]


def fisher_exact_wilcoxon(
    spike_collection, event1, event2, event_length, event3=None, baseline_window=None, offset=0, exclude_offset=False
):
    """
    Calculates fisher's exact test: contigency matrix is made up of number of significant units from wilcoxon signed
    rank test of baseline_window vs event vs non-significant units for event1 and event2.

    Args(7 total, 3 required):
        event1: str, event in event_dict
        event2: str, event in event_dict
        event3: str, default=None, if not none, event type to be used as baseline in wilcoxon signed rank sum test
        event_length: int, length (s) of event to be used
        baseline_window: int, default=None, if not None, length (s) of baseline window immediately prior to event onset
        offset: int, adjusts end of baseline by offset(s) from onset of behavior i.e. offset=2 adds first 2s of
            event firing rates onto baseline, offest=-2 removes firing rate data from baseline averages
        exclude_offset: Boolean, default=False, if true excludes time prior to onset and before offset in event avgs,
            if false, time between onset and offset are included in event averages

    Returns (3):
        odds_ratio: float, fisher's exact test results
        p_value: float, p value
        contingency_matrix: np.array (d=2x2):
            [[event1 significant units, event1 non-significnat units],
            [event2 significant units, event2 non-significant units]]
    """
    if (event3 is None) & (baseline_window is None):
        print("Function needs a baseline event or window")
        print("Please set either baseline_window or event3 to a value")
    if (event3 is not None) & (baseline_window is not None):
        print("Function can only handle one baseline for comparison.")
        print("baseline_window OR event3 must equal None")
    if event3 is None:
        df1 = wilcoxon_collection(
            spike_collection, event1, event_length, baseline_window, offset, exclude_offset, plot=False
        )
        df2 = wilcoxon_collection(
            spike_collection, event2, event_length, baseline_window, offset, exclude_offset, plot=False
        )
    else:
        df1 = wilcoxon_event1v2_collection(spike_collection, event1, event3, event_length, plot=False)
        df2 = wilcoxon_event1v2_collection(spike_collection, event2, event3, event_length, plot=False)
    row1 = [(df1["p value"] < 0.05).sum(), (df1["p value"] > 0.05).sum()]
    row2 = [(df2["p value"] < 0.05).sum(), (df2["p value"] > 0.05).sum()]
    contingency_matrix = pd.DataFrame(
        [row1, row2], index=[f"{event1}", f"{event2}"], columns=["Significant", "Non-Significant"]
    )
    odds_ratio, p_value = fisher_exact(contingency_matrix)
    return odds_ratio, p_value, contingency_matrix


def baseline_v_event_plot(spike_collection, master_df, event, event_length, baseline_window, offset):
    """
    plots event triggered average firing rates for units with significant
    wilcoxon signed rank tests (p value <0.05) for event v baseline window.

    Args(4 total, 4 required):
        event: str, event type of which ehpys snippets happen during
        event_length: float, length (s) of events used by padding with post
            event time or trimming events all to event_length (s) long used
        baseline_window: int, default=0, seconds prior to start of event
        offset: int, adjusts end of baseline by offset(s) from onset of
            behavior such that offset=2 adds the first two seconds of event
            data into baseline while offest=-2 removes them from baseline
            averages

    Returns:
        none
    """
    all_units_to_plot = []
    for recording in spike_collection.recordings:
        wilcoxon_df = master_df[master_df["Recording"] == recording.name]
        recording_units = wilcoxon_df[wilcoxon_df["p value"] < 0.07]["original unit id"].tolist()
        all_units_to_plot.extend([(recording, unit) for unit in recording_units])

    no_plots = len(all_units_to_plot)
    height_fig = math.ceil(no_plots / 3)

    plt.figure(figsize=(20, 4 * height_fig))
    i = 1

    for recording, unit in all_units_to_plot:
        wilcoxon_df = master_df[master_df["Recording"] == recording.name]
        unit_event_firing_rates = recording.unit_event_firing_rates(event, event_length, baseline_window, 0)

        mean_arr = np.mean(unit_event_firing_rates[unit], axis=0)
        sem_arr = sem(unit_event_firing_rates[unit], axis=0)
        p_value = wilcoxon_df.loc[wilcoxon_df["original unit id"] == unit, "p value"].values[0]

        x = np.linspace(start=-baseline_window, stop=event_length, num=len(mean_arr))
        plt.subplot(height_fig, 3, i)
        plt.plot(x, mean_arr, c="b")

        if offset != 0:
            plt.axvline(x=offset, color="b", linestyle="--")
        plt.axvline(x=0, color="r", linestyle="--")
        plt.fill_between(x, mean_arr - sem_arr, mean_arr + sem_arr, alpha=0.2)
        plt.title(f"{recording.name} - Unit {unit} (p={p_value:.3f})")
        i += 1
    # plt.subplots_adjust(top=1.1)
    plt.suptitle(f"{event_length}s {event} vs {baseline_window}s baseline", y=1, fontsize=20)
    plt.tight_layout()
    plt.show()


def event1v2_plot(spike_collection, master_df, event1, event2, event_length, pre_window):
    """
    plots event triggered average firing rates for units with significant wilcoxon
    signed rank sums (p value < 0.05) for event1 vs event2

        Args(5 total, 5 required):
        master_df: dataframe, return of event_v_event_collection function
        event1: str, event type 1
        event2: str, event type 2
        event_length: int, length (s) of events
        pre_window: int, time (s) prior to event onset to be plotted

    Returns:
        none
    """

    all_units_to_plot = []
    for recording in spike_collection.recordings:
        wilcoxon_df = master_df[master_df["Recording"] == recording.name]
        recording_units = wilcoxon_df[wilcoxon_df["p value"] < 0.05]["original unit id"].tolist()
        all_units_to_plot.extend([(recording, unit) for unit in recording_units])

    no_plots = len(all_units_to_plot)
    height_fig = math.ceil(no_plots / 3)

    fig = plt.figure(figsize=(20, 4 * height_fig))
    fig.suptitle(f"{event1} vs {event2} ({event_length}s)", y=1.02)

    for idx, (recording, unit) in enumerate(all_units_to_plot, 1):
        wilcoxon_df = master_df[master_df["Recording"] == recording.name]
        unit_event1_firing_rates = recording.unit_event_firing_rates(event1, event_length, pre_window, 0)
        unit_event2_firing_rates = recording.unit_event_firing_rates(event2, event_length, pre_window, 0)

        mean1_arr = np.mean(unit_event1_firing_rates[unit], axis=0)
        sem1_arr = sem(unit_event1_firing_rates[unit], axis=0)
        mean2_arr = np.mean(unit_event2_firing_rates[unit], axis=0)
        sem2_arr = sem(unit_event2_firing_rates[unit], axis=0)
        p_value = wilcoxon_df.loc[wilcoxon_df["original unit id"] == unit, "p value"].values[0]

        x = np.linspace(start=-pre_window, stop=event_length, num=len(mean1_arr))
        ax = fig.add_subplot(height_fig, 3, idx)
        ax.plot(x, mean1_arr, c="b", label=event1)
        ax.fill_between(x, mean1_arr - sem1_arr, mean1_arr + sem1_arr, alpha=0.2)
        ax.plot(x, mean2_arr, c="k", label=event2)
        ax.fill_between(x, mean2_arr - sem2_arr, mean2_arr + sem2_arr, alpha=0.2, color="k")
        ax.axvline(x=0, color="r", linestyle="--")
        ax.set_title(f"{recording.name} - Unit {unit} (p={p_value:.3f})")
        ax.legend()

    plt.tight_layout()
    plt.show()


def wilcoxon_unit(recording, unit_id, events, event_length, baseline_window, offset, exclude_offset=False):
    """
    plots event triggered average firing rates for units with significant
    wilcoxon signed rank tests (p value <0.05) for event v baseline window.

    Args(4 total, 4 required):
        events: list of str, event types of which ehpys snippets happen during
        event_length: float, length (s) of events used by padding with post
            event time or trimming events all to event_length (s) long used
        baseline_window: int, default=0, seconds prior to start of event
        offset: int, adjusts end of baseline by offset(s) from onset of
            behavior such that offset=2 adds the first two seconds of event
            data into baseline while offest=-2 removes them from baseline
            averages

    Returns:
        none
    """
    no_plots = len(events)
    height_fig = math.ceil(no_plots / 2)
    i = 1
    plt.figure(figsize=(15, 4 * height_fig))
    for event in events:
        wilcox_df = wilcoxon_rec(recording, event, event_length, baseline_window, offset, exclude_offset)
        unit_event_firing_rates = recording.unit_event_firing_rates(event, event_length, baseline_window)
        mean_arr = np.mean(unit_event_firing_rates[unit_id], axis=0)
        sem_arr = sem(unit_event_firing_rates[unit_id], axis=0)
        p_value = wilcox_df["p value"].values[0]
        x = np.linspace(start=-baseline_window, stop=event_length, num=len(mean_arr))
        plt.subplot(height_fig, 2, i)
        plt.plot(x, mean_arr, c="b")
        if offset != 0:
            plt.axvline(x=offset, color="b", linestyle="--")
        plt.axvline(x=0, color="r", linestyle="--")
        plt.fill_between(x, mean_arr - sem_arr, mean_arr + sem_arr, alpha=0.2)
        plt.title(f"{event}: p={p_value}")
        i += 1
    plt.suptitle(f"{recording.name}: {unit_id}")
    plt.show()


def bootstrap(spike_collection, events, event_length, pre_window=0, num_perm=2000):
    results_dict = {}
    for recording in spike_collection.recordings:
        results_dict[recording.name] = {}
        for event in events:
            unit_dict = bootstrap_recording(recording, event, event_length, pre_window, num_perm)
            results_dict[recording.name][event] = unit_dict

    rows = []
    # Iterate through the nested structure
    for recording, event_dict in results_dict.items():
        # Get all units from all events in this recording
        all_units = set()
        for event_data in event_dict.values():
            all_units.update(event_data.keys())
        # Create a row for each unit
        for unit in all_units:
            row = {"original_unit_id": unit, "recording": recording}
            # Add values for each event
            for event, unit_dict in event_dict.items():
                row[event] = unit_dict.get(unit, None)  # Use None if unit doesn't have this event
            rows.append(row)
    # Create DataFrame
    df = pd.DataFrame(rows)
    return df


def bootstrap_recording(recording, event, event_length, pre_window, num_perm):
    firing_rates = recording.unit_firing_rates
    unit_dict = {}
    for unit, firing_rate in firing_rates.items():
        shuffled_units = np.array([np.random.permutation(firing_rate) for _ in range(num_perm)])
        # Get all event snippets at once
        all_shuffled_snippets = np.array(
            [
                recording.event_snippets(event, shuffled_unit, event_length, pre_window)
                for shuffled_unit in shuffled_units
            ]
        )
        # Calculate means all at once
        shuffle_means = np.mean(np.mean(all_shuffled_snippets, axis=1), axis=1)
        low_end = np.percentile(shuffle_means, 2.5)
        high_end = np.percentile(shuffle_means, 97.5)
        event_firing_rates = recording.event_snippets(event, firing_rate, event_length, pre_window)
        avg_event_firing_rate = np.mean(np.mean(event_firing_rates, axis=0))
        if avg_event_firing_rate > high_end:
            unit_dict[unit] = "significantly increased"
        elif avg_event_firing_rate < low_end:
            unit_dict[unit] = "significantly decreased"
        else:
            unit_dict[unit] = "not significantly changed"
    return unit_dict


def plot_raster(spike_collection, event, event_length, pre_window, global_timebin=1000):

    zscore_df = normalization.zscore_global(
        spike_collection, event, event_length, pre_window, global_timebin, plot=False
    )
    zscore_df = zscore_df.drop(columns=["Recording", "Event", "Subject", "original unit id"])
    # Get sorting indices (in descending order, hence the [::-1])
    zscore_array = zscore_df.to_numpy()

    row_means = np.mean(zscore_array, axis=1)
    sort_indices = np.argsort(row_means)[::-1]

    # Reorder the data
    sorted_zscore = zscore_array[sort_indices]
    timebin_s = spike_collection.timebin / 1000  # Convert timebin from ms to seconds
    total_bins = sorted_zscore.shape[1]
    time_axis = np.linspace(-pre_window, (event_length - timebin_s), total_bins)

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 10))

    # Plot heatmap
    im = ax.imshow(
        sorted_zscore, aspect="auto", cmap="viridis", extent=[time_axis[0], time_axis[-1], sorted_zscore.shape[0], 0]
    )

    # Add event rectangle (from 0 to event_length in seconds)
    import matplotlib.patches as patches

    rect = patches.Rectangle(
        (0, 0), event_length / 1000, sorted_zscore.shape[0], linewidth=2, edgecolor="red", facecolor="none", alpha=0.5
    )
    ax.add_patch(rect)

    # Customize plot
    plt.colorbar(im, label="Z-score")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Units (sorted by mean activity)")
    ax.set_title("Neural Activity Aligned to Event Onset")

    # Add vertical line at event onset
    ax.axvline(x=0, color="red", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.show()
