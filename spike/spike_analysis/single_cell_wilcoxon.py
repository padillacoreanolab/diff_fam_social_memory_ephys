import numpy as np
from statistics import StatisticsError
import pandas as pd
from scipy.stats import sem, ranksums, wilcoxon
import math
import matplotlib.pyplot as plt


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


def event_avgs(event1_firing_rates, event2_firing_rates):
    unit_averages = {}
    for unit in event1_firing_rates.keys():
        try:
            event1_averages = [np.nanmean(event) for event in event1_firing_rates[unit]]
            event2_averages = [np.nanmean(event) for event in event2_firing_rates[unit]]
            unit_averages[unit] = [np.array(event1_averages), np.array(event2_averages)]
        except StatisticsError:
            print(f"Unit {unit} has too few spikes")
        return unit_averages


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
    wilcoxon_stats = {}
    for unit in unit_averages.keys():
        if not np.isnan(unit_averages[unit][:]).any():  # Check if data is valid before running Wilcoxon
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
        if not np.isnan(unit_averages[unit][:]).any():
            results = ranksums(unit_averages[unit][0], unit_averages[unit][1])
            wilcoxon_stats[unit] = {"Wilcoxon Stat": results.statistic, "p value": results.pvalue}
        else:
            wilcoxon_stats[unit] = {"Wilcoxon Stat": np.nan, "p value": np.nan}
    wilcox_df = dict_to_df(wilcoxon_stats)
    return wilcox_df


def baseline_v_event_plot(recording, unit_event_firing_rates, master_df, event, event_length, baseline_window, offset):
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
    wilcoxon_df = master_df[master_df["Recording"] == recording]
    units_to_plot = []
    for unit in wilcoxon_df["original unit id"].tolist():
        if wilcoxon_df.loc[wilcoxon_df["original unit id"] == unit, "p value"].values[0] < 0.07:
            units_to_plot.append(unit)
    no_plots = len(units_to_plot)
    height_fig = math.ceil(no_plots / 3)
    i = 1
    plt.figure(figsize=(20, 4 * height_fig))
    for unit in units_to_plot:
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
        plt.title(f"Unit {unit} Average (p={p_value})")
        i += 1
    plt.suptitle(f"{recording}: " + f"{event_length}s {event} vs {baseline_window}s baseline")
    plt.show()


def event_v_event_plot(
    recording, unit_event1_firing_rates, unit_event2_firing_rates, master_df, event1, event2, event_length, pre_window
):
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
    wilcoxon_df = master_df[master_df["Recording"] == recording]
    units_to_plot = []
    for unit in wilcoxon_df["original unit id"].tolist():
        if wilcoxon_df.loc[wilcoxon_df["original unit id"] == unit, "p value"].values[0] < 0.05:
            units_to_plot.append(unit)
    no_plots = len(units_to_plot)
    height_fig = math.ceil(no_plots / 3)
    i = 1
    plt.figure(figsize=(20, 4 * height_fig))
    for unit in units_to_plot:
        mean1_arr = np.mean(unit_event1_firing_rates[unit], axis=0)
        sem1_arr = sem(unit_event1_firing_rates[unit], axis=0)
        mean2_arr = np.mean(unit_event2_firing_rates[unit], axis=0)
        sem2_arr = sem(unit_event2_firing_rates[unit], axis=0)
        p_value = wilcoxon_df.loc[wilcoxon_df["original unit id"] == unit, "p value"].values[0]
        x = np.linspace(start=-pre_window, stop=event_length, num=len(mean1_arr))
        plt.subplot(height_fig, 3, i)
        plt.plot(x, mean1_arr, c="b", label=event1)
        plt.fill_between(x, mean1_arr - sem1_arr, mean1_arr + sem1_arr, alpha=0.2)
        plt.plot(x, mean2_arr, c="k", label=event2)
        plt.fill_between(x, mean2_arr - sem2_arr, mean2_arr + sem2_arr, alpha=0.2, color="k")
        plt.axvline(x=0, color="r", linestyle="--")
        plt.title(f"Unit {unit} Average (p={p_value})")
        plt.legend()
        i += 1
    plt.suptitle(f"{recording}: " + f"{event1} vs {event2} ({event_length}s)")
    plt.show()


# TO DO FINISH THIS FUNCTION
def wilcox_baseline_v_event_unit(
    self, recording_name, unit_id, events, event_length, baseline_window, offset, exclude_offset=False
):
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
    recording = self.ephyscollection.get_by_name(recording_name)
    for event in events:
        temp_master_df = self.wilcox_baseline_v_event_collection(
            event, event_length, baseline_window, offset, exclude_offset, plot=False, save=False
        )
        master_df = temp_master_df[
            (temp_master_df["Recording"] == recording_name) & (temp_master_df["original unit id"] == unit_id)
        ]
        master_df = master_df.reset_index()
        unit_event_firing_rates = self.__get_unit_event_firing_rates__(
            recording, event, event_length, baseline_window, 0
        )
        mean_arr = np.mean(unit_event_firing_rates[unit_id], axis=0)
        sem_arr = sem(unit_event_firing_rates[unit_id], axis=0)
        p_value = master_df["p value"].values[0]
        x = np.linspace(start=-baseline_window, stop=event_length, num=len(mean_arr))
        plt.subplot(height_fig, 2, i)
        plt.plot(x, mean_arr, c="b")
        if offset != 0:
            plt.axvline(x=offset, color="b", linestyle="--")
        plt.axvline(x=0, color="r", linestyle="--")
        plt.fill_between(x, mean_arr - sem_arr, mean_arr + sem_arr, alpha=0.2)
        plt.title(f"{event}: p={p_value}")
        i += 1
    plt.suptitle(f"{recording_name}: {unit_id}")
    plt.show()
