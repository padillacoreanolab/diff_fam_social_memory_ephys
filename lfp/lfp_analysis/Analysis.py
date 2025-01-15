import numpy as np
from scipy import stats
import math
import matplotlib.pyplot as plt
from itertools import permutations
import seaborn as sns
from bidict import bidict


class LFPAnalysis:

    def __init__(self, lfp_objects, sampling_rate=20000, down_sample=1000, freq_window=0.5, freq_smooth_win=1):
        """
        lfp_objects: list of lfprecording instances
        sampling_rate: int, default=20000, sampling rate of ephys device in Hz
        down_sample: int, default=1000, down sample rate of LFP traces in Hz
        freq_winddow: float, default=0.5, timebin in seconds for frequency based arrays such as
            power, coherence, and granger
        freq_smooth_win: float, default=1, size of smoothing window in sec to calculate power,
            coherence, and granger causality
        """
        self.sampling_rate = sampling_rate
        self.down_sample = down_sample
        self.collection = lfp_objects
        self.power_frequencies = lfp_objects[0].frequencies
        self.coherence_frequencies = lfp_objects[0].frequencies
        self.granger_frequencies = lfp_objects[0].frequencies
        self.brain_region_dict = lfp_objects[0].brain_region_dict
        self.freq_window = freq_window
        self.freq_smooth_win = freq_smooth_win
        self.all_set()
        self.coherence_pairs_dict = self._create_coherence_bidict()

    def all_set(self):
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
        for i in range(len(self.collection)):
            recording = self.collection[i]
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
                missing_subject.append(recording.recording_name)
        if len(missing_events) > 0:
            print("These recordings are missing event dictionaries:")
            print(f"{missing_events}")
            is_good = False
        else:
            if not behavior_dicts_same:
                print("Your event dictionary keys are different across recordings.")
                print("Please double check them:")
                for (
                    recording_name,
                    recording,
                ) in self.ephyscollection.collection.items():
                    print(recording_name, "keys:", recording.behavior_dict.keys())
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

    def __get_event_averages__(
        self,
        recording,
        event,
        mode,
        event_len,
        pre_window,
        post_window,
    ):
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
        event_averages = []
        pre_window = math.ceil(pre_window * 1000)
        post_window = math.ceil(post_window * 1000)
        freq_timebin = self.freq_window * 1000
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
                event_average = np.nanmean(event_snippet, axis=0)
                event_averages.append(event_average)
        return event_averages

    def plot_average_events(self, event_averages, mode, regions=None):
        if mode == "power":
            self.plot_power_averages(event_averages, regions)
        if mode == "coherence":
            self.plot_coherence_averages(event_averages, regions)
        if mode == "granger":
            self.plot_granger_averages(event_averages, regions)

    def plot_power_averages(self, event_averages, regions=None):
        if regions is None:
            regions = self.brain_region_dict.keys()
        for region in regions:
            plt.figure(figsize=(10, 5))
            for event, averages in event_averages.items():
                # averages = [trials, f, b]
                averages = event_averages[event]
                event_average = np.nanmean(averages, axis=0)
                # event_average = [f,b]; average across all trials
                # calculate sem for the trial average
                event_sem = stats.sem(averages, axis=0, nan_policy="omit")
                region_index = self.brain_region_dict[region]
                # pick only the region of interest
                y = event_average[1:100, region_index]
                y_sem = event_sem[1:100, region_index]
                x = self.power_frequencies[1:100]
                (line,) = plt.plot(x, y, label=event)
                plt.fill_between(x, y - y_sem, y + y_sem, alpha=0.2, color=line.get_color())
            ymin, ymax = plt.ylim()
            plt.axvline(x=12, color="gray", linestyle="--", linewidth=0.5)
            plt.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)
            plt.fill_betweenx(y=np.linspace(ymin, ymax, 80), x1=4, x2=12, color="red", alpha=0.1)
            plt.ylim(ymin, ymax)
            plt.title(f"{region} power")
            plt.legend()
            plt.show()

    def plot_coherence_averages(self, event_averages, regions=None):
        if regions is not None:
            pairs_indices = []
            for region in regions:
                try:
                    pairs_index = self.coherence_pairs_dict[frozenset({region[0], region[1]})]
                except KeyError:
                    pairs_index = self.coherence_pairs_dict[frozenset({region[0], region[1]})]
                pairs_indices.append(pairs_index)
        if regions is None:
            regions = list(self.coherence_pairs_dict.values())
        for i in range(len(regions)):
            for event, averages in event_averages.items():
                # averages = [trials, f, b, b]
                first_region, second_region = list(regions[i])
                first_region_name = self.brain_region_dict.inverse[first_region]
                second_region_name = self.brain_region_dict.inverse[second_region]
                averages = event_averages[event]
                event_average = np.nanmean(averages, axis=0)
                # event_average = [f, b, b]; average across all trials
                # calculate sem for the trial average
                event_sem = stats.sem(averages, axis=0, nan_policy="omit")
                # pick only the region of interest
                y_sem = event_sem[1:100, first_region, second_region]
                y = event_average[1:100, first_region, second_region]
                x = self.coherence_frequencies[1:100]
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

    def plot_granger_averages(self, event_averages, regions=None):
        if regions is not None:
            pair_indices = []
            for region in regions:
                first_index = self.brain_region_dict[region[0]]
                second_index = self.brain_region_dict[region[1]]
                pair_indices.append([first_index, second_index])
        if regions is None:
            pair_indices = list(permutations(range(len(self.brain_regions)), 2))
            regions = []
            for pair in pair_indices:
                regions.append([self.brain_regions[pair[0]], self.brain_regions[pair[1]]])
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
                y_sem = event_sem[pair_indices[i][0], pair_indices[i][1], 1:100]
                y = event_average[pair_indices[i][0], pair_indices[i][1], 1:100]
                x = self.coherence_frequencies[1:80]
                (line,) = plt.plot(x, y, label=event)
                # plt.fill_between(x, y - y_sem, y +y_sem, color = line.get_color(), alpha = 0.2)
            ymin, ymax = plt.ylim()
            plt.axvline(x=12, color="gray", linestyle="--", linewidth=0.5)
            plt.axvline(x=4, color="gray", linestyle="--", linewidth=0.5)
            plt.fill_betweenx(y=np.linspace(ymin, ymax, 80), x1=4, x2=12, color="red", alpha=0.1)
            plt.ylim(ymin, ymax)
            plt.title(f"Granger causality: {region[0]} to {region[1]}")
            plt.legend()
            plt.show()

    def plot_granger_heatmap(self, events, freq, baseline=None, event_len=None):
        event_grangers = self.average_events(events, mode="granger", baseline=baseline, event_len=event_len, plot=False)
        for event in events:
            event_granger = event_grangers[event]
            # event_granger = [trial, b, b, f]
            # calculate average granger per trial
            avg_granger = np.nanmean(event_granger, axis=0)
            # trim array for only the freq band we are interested in
            freq_granger = avg_granger[:, :, freq[0] : freq[1]]
            # average for that freq band
            avg_freq = np.nanmean(freq_granger, axis=2)
            sns.heatmap(
                avg_freq, xticklabels=self.brain_regions, yticklabels=self.brain_regions, annot=True, cmap="viridis"
            )
            plt.title(f"{event} Granger Causality from {freq[0]}Hz to {freq[1]}Hz")
            plt.show()

    def average_events(self, events, mode, baseline=None, event_len=None, pre_window=0, post_window=0, plot=False):
        """
        Calculates average event measurement (power, coherence, or granger) per recording then
        calculates global averages across all recordings from recording averages (to account for
        differences in event numbers per recording)
        """
        event_averages_dict = {}
        for event in events:
            recording_averages = []
            for recording in self.collection:
                event_averages = self.__get_event_averages__(recording, event, mode, event_len, pre_window, post_window)
                if baseline is not None:
                    adj_averages = self.__baseline_diff__(
                        recording, event_averages, baseline, mode, event_len, pre_window=0, post_window=0
                    )
                    recording_averages = recording_averages + adj_averages
                else:
                    recording_averages = recording_averages + event_averages
            # recording_averages = [trials, b, f] or [trials, b, b, f]
            event_averages_dict[event] = recording_averages
        if plot:
            self.plot_average_events(event_averages_dict, mode)
        return event_averages_dict

    def __baseline_diff__(self, recording, event_averages, baseline, mode, event_len, pre_window, post_window):
        baseline_averages = self.__get_event_averages__(recording, baseline, mode, event_len, pre_window, post_window)
        baseline_recording = np.nanmean(np.array(baseline_averages), axis=0)
        adj_averages = []
        for i in range(len(event_averages)):
            adj_average = ((event_averages[i] - baseline_recording) / (baseline_recording + 0.00001)) * 100
            adj_averages.append(adj_average)
        return adj_averages

    def __get_event_snippets__(
        self,
        recording,
        event,
        mode,
        event_len,
        pre_window,
        post_window,
    ):
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
        freq_timebin = self.freq_window * 1000
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
