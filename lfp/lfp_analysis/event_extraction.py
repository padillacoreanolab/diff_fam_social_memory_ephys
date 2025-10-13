import numpy as np
from scipy import stats
import math
from itertools import permutations
from bidict import bidict
from collections import defaultdict
from matplotlib import cm
from itertools import combinations
from lfp.lfp_analysis.LFP_collection import LFPCollection
from lfp.lfp_analysis.LFP_recording import LFPRecording
import lfp.lfp_analysis.plotting as lfplt


def all_set(collection):
    """
    double checks that all lfp objects in the collection have
    the attributes: subject & event_dict assigned to them and that
    each event_dict has the same keys.

    Prints statements telling user which recordings are missing subjects
    or event_dicts.
    Prints event_dict.keys() if they are not the same.
    Prints "All set to analyze"
    """
    is_first = True
    is_good = True
    missing_events = []
    missing_subject = []
    event_dicts_same = True
    for i in range(len(collection.recordings)):
        recording = collection.recordings[i]
        if not hasattr(recording, "event_dict"):
            missing_events.append(recording)
        else:
            if is_first:
                last_recording_events = recording.event_dict.keys()
                is_first = False
            else:
                if recording.event_dict.keys() != last_recording_events:
                    event_dicts_same = False
        if not hasattr(recording, "subject"):
            missing_subject.append(recording.name)
    if len(missing_events) > 0:
        print("These recordings are missing event dictionaries:")
        print(f"{missing_events}")
        is_good = False
    else:
        if not event_dicts_same:
            print("Your event dictionary keys are different across recordings.")
            print("Please double check them:")
            for recording in collection.recordings:
                print(recording.name, "keys:", recording.event_dict.keys())
    if len(missing_subject) > 0:
        print(f"These recordings are missing subjects: {missing_subject}")
        is_good = False
    if is_good:
        print("All set to analyze")


def get_events(recording, event, mode, event_len, pre_window, post_window, average=True):
    """
    takes snippets of power, coherence, or causality for events
    optional pre-event and post-event windows (s) may be included
    all events can also be of equal length by extending
    snippet lengths to the longest event

    Args (6 total, 4 required):
        recording: LFP object instance, recording to get snippets
        event: str, event type of which ehpys snippets happen during
        mode: str, {'power', 'coherence', gangers'}, type of measurement to get event
            snippets for
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
        events = recording.event_dict[event]

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
        whole_recording = recording.granger
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


def event_difference(
    lfp_collection,
    event1,
    event2,
    mode,
    baseline1=None,
    baseline2=None,
    event_len=None,
    pre_window=0,
    post_window=0,
    plot=False,
    regions=None,
    freq_range=None,
):
    diff_dict = {}
    n_regions = len(lfp_collection.brain_region_dict.keys())
    if mode == "power":
        event1_averages = np.zeros([len(lfp_collection.recordings), 500, n_regions])
        event2_averages = np.zeros([len(lfp_collection.recordings), 500, n_regions])
    else:
        event1_averages = np.zeros([len(lfp_collection.recordings), 500, n_regions, n_regions])
        event2_averages = np.zeros([len(lfp_collection.recordings), 500, n_regions, n_regions])
    for i in range(len(lfp_collection.recordings)):
        recording = lfp_collection.recordings[i]
        event1_avg = __get_events__(recording, event1, mode, event_len, pre_window, post_window)
        event2_avg = __get_events__(recording, event2, mode, event_len, pre_window, post_window)
        if baseline1 is not None:
            event1_avg = __baseline_diff__(
                recording, event1_avg, baseline1, mode, event_len, pre_window=0, post_window=0, average=True
            )
        if baseline2 is not None:
            event2_avg = __baseline_diff__(
                recording, event2_avg, baseline2, mode, event_len, pre_window=0, post_window=0, average=True
            )
        # recording_averages = [trials, b, f] or [trials, b, b, f]
<<<<<<< Updated upstream
        event1_avg = np.mean(np.array(event1_avg), axis=0)
        event2_avg = np.mean(np.array(event2_avg), axis=0)
        event1_averages[i, ...] = event1_avg
        event2_averages[i, ...] = event2_avg
    event_diff = (event1_averages - event2_averages) / (event1_averages + event2_averages) * 100
    diff_dict[f"{event1} vs {event2}"] = event_diff
=======
        event1_avg = np.nanmean(np.array(event1_avg), axis = 0)
        event2_avg = np.nanmean(np.array(event2_avg), axis = 0)
        event1_averages[i,...] = event1_avg
        event2_averages[i,...] = event2_avg
    event_diff = (event1_averages - event2_averages) / (event1_averages + event2_averages)*100
    diff_dict[f'{event1} vs {event2}'] = event_diff
>>>>>>> Stashed changes
    if plot:
        plot_average_events(lfp_collection, diff_dict, mode, regions, freq_range)
    return diff_dict


def __baseline_diff__(recording, event_averages, baseline, mode, event_len, pre_window, post_window, average):
    baseline_averages = get_events(recording, baseline, mode, event_len, pre_window, post_window, average=average)
    # average = trial, freq, regions
    # not average = trial, time, freq, regions
    if not average:
        baseline_recording = np.nanmean(np.nanmean(np.array(baseline_averages), axis=0), axis=0)
    if average:
        baseline_recording = np.nanmean(np.array(baseline_averages), axis=0)
    adj_averages = []
    for i in range(len(event_averages)):
        adj_average = ((event_averages[i] - baseline_recording) / (baseline_recording + 0.00001)) * 100
        adj_averages.append(adj_average)
    return adj_averages


def average_events(
    lfp_collection,
    events,
    mode,
    baseline=None,
    event_len=None,
    pre_window=0,
    post_window=0,
    plot=False,
    regions=None,
    freq_range=None,
):
    """
    Calculates average event measurement (power, coherence, or granger) per recording then
    calculates global averages across all recordings from recording averages (to account for
    differences in event numbers per recording)
    """
    event_averages_dict = {}
    if (not isinstance(baseline, list)) and (baseline is not None):
        baseline = [baseline]
    if isinstance(lfp_collection, LFPCollection):
        recordings = lfp_collection.recordings
    if isinstance(lfp_collection, LFPRecording):
        recordings = [lfp_collection]
    if baseline is not None:
        if (len(events) != len(baseline)) and (len(baseline) == 1):
            baseline = baseline * len(events)
    for i in range(len(events)):
        recording_averages = []
        for recording in recordings:
            event_averages = get_events(recording, events[i], mode, event_len, pre_window, post_window, average=True)

            if baseline is not None:
                adj_averages = __baseline_diff__(
                    recording, event_averages, baseline[i], mode, event_len, pre_window=0, post_window=0, average=True
                )
<<<<<<< Updated upstream
                rec_event_average = np.mean(np.array(adj_averages), axis=0)
                recording_averages.append(rec_event_average)
            else:
                rec_event_average = np.mean(np.array(event_averages), axis=0)
=======
                rec_event_average = np.nanmean(np.array(adj_averages), axis = 0)
                recording_averages.append(rec_event_average)
            else:
                rec_event_average = np.nanmean(np.array(event_averages), axis = 0)
>>>>>>> Stashed changes
                recording_averages.append(rec_event_average)

        # recording_averages = [trials, b, f] or [trials, b, b, f]

        event_averages_dict[events[i]] = recording_averages
    if plot:
        lfplt.plot_average_events(lfp_collection, event_averages_dict, mode, regions, freq_range)
    return event_averages_dict
<<<<<<< Updated upstream
=======

def baselined_events(
    lfp_collection, events, mode, baseline=None, event_len=None, pre_window=0, post_window=0, plot=False, regions = None, freq_range = None
):
    """
    Calculates average event measurement (power, coherence, or granger) per recording then
    calculates global averages across all recordings from recording averages (to account for
    differences in event numbers per recording)
    """
    event_averages_dict = {}
    if (not isinstance(baseline, list)) and (baseline is not None): 
        baseline = [baseline]
    if isinstance(lfp_collection , LFPCollection):
        recordings = lfp_collection.recordings
    if isinstance(lfp_collection , LFPRecording):
        recordings = [lfp_collection]
    if baseline is not None:
        if (len(events) != len(baseline)) and (len(baseline) == 1):
            baseline = baseline * len(events)
    for i in range(len(events)):
        recording_averages = []
        for recording in recordings:
            event_averages = get_events(recording, events[i], mode, event_len, pre_window, post_window, average = True)
            
            if baseline is not None:
                adj_averages = __baseline_diff__(
                    recording, event_averages, baseline[i], mode, event_len, pre_window=0, post_window=0, average = True
                )
           
                recording_averages.append(adj_averages)
            else:
  
                recording_averages.append(event_averages)
            
        # recording_averages = [trials, b, f] or [trials, b, b, f]
        rec_avg_array = np.moveaxis((np.array(recording_averages)), 1, -1)
        
        event_averages_dict[events[i]] = rec_avg_array
    # if plot:
    #     lfplt.plot_average_events(lfp_collection, event_averages_dict, mode, regions, freq_range)
    return event_averages_dict
>>>>>>> Stashed changes
