import numpy as np
import spike_analysis.single_cell_wilcoxon as wilcox
import spike_analysis.firing_rate_calculations as fr
import pandas as pd
from scipy.stats import sem, ranksums, fisher_exact, wilcoxon


class SpikeAnalysis:
    """
    A class for ephys statistics done on multiple event types for multiple
    recordings. Each recording needs a subject (recording.subject) and
    event_dict (recording.event_dict) attribute assigned to it before analysis.
    event_dict is such that the keys are event type names (str)
    and values are np.arrays [[start (ms), stop(ms)]..]
    of start and stop times for each event.

    This class can do statistics calculations on firing rates for events:
    1a) wilcoxon signed rank tests for baseline vs event
    1b) fishers exact tests on units that have significant wilcoxon
        signed rank results for baseline vs event1 and baseline vs event2
    1c) wilcoxon signed rank sum tests for event1 vs event2
    2) zscored average even firing rates
    3) PCA embeddings on any number of event types

    All stats can be exported as excel sheets in the parent
    directory of the collection.

    Attributes:
        timebin: int, bin size (in ms) for spike train
            and firing rate arrays
        ignore_freq: int, default=0.1, frequency in Hz, any good unit that fires
            < than ignore_freq will be excluded from analysis
        smoothing_window: int, default=None, window length (ms) used
            to calculate firing rates, if None, then no smoothing occurs

    Methods:
        wilcox_baseline_v_event_collection: Runs a wilcoxon signed rank test on all good units of
                    all recordings in the collection on the given event's firing rate versus the given
                    baseline window. Default plots results. Creates and returns a dataframe with rows for each unit
                    and columns representing Wilcoxon stats, p values, orginal unit ids, recording,
                    subject and the event + baselien given. Dataframe is saved if save mode is True
                    in the collections attribute wilcox_dfs dictionary, key is
                    '{event} vs {baseline_window}second baseline'. Option to save this dataframe for export.
        fishers_exact_wilcox: Calculates and returns odds ratio, p value, and contingency matrix using fisher's exact test
                    The contigency matrix is made up of number of significant units
                    (from wilcoxon signed rank test of baseline_window vs event) vs non-significant
                    units for event1 and event12. Option to save output stats for export.
        wilcox_event_v_event_collection: Runs a wilcoxon signed rank test on all good units of
                    all recordings in the collection on the given event's firing rate versus
                    another given event's firing rate. Default plots results. Creates and returns a dataframe with
                    rows for each unit and columns representing Wilcoxon stats, p values, orginal unit ids,
                    recording, subject and the events given. Option to save dataframe in the collections
                    wilcox_dfs dictionary, key is '{event1 } vs {event2} ({event_length}s)' for export.
        zscore_collection: calculates z-scored event average firing rates for all recordings in the collection.
                    Default plots the results. Option to save for export a dataframe of all zscored event firing rates
                    with columns for original unit id, recording name, and subject as a value in
                    zscored_event dictionary attribute of the colleciton.
                    '{event_length}s {event} vs {baseline_window}s baseline' is the key to the dataframe
        PCA_trajectories: calculates and default plots a PCA matrix where each data point represents a timebin.
                    PCA space is calculated from a matrix of all units and all timebins
                    from every type of event in event dict or events in events.
                    PCA matrix can be saved for export where key is events list or 'all'
        export: saves all saved stats as excel files in the same parent dictory with which the ephys
                    collection was created from

    Private Methods (aka one's you do not need to run):
        __all_set__: checks that each recording the collection has a subject and an event_dict and
                    that all event_dicts have the same event types
        __get_whole_spiketrain__: assigns spiketrians as attribute for each recording, spiketrains are
                    np. arrays where each element is the number of spikes per timebin
        __get_unit_spiketrains__: Creates a dictionary and assigns it as recording.unit_spiketrains
                    for each recording in the collection where keys are 'good' unit ids (int) that
                    reach a threshold frequency, values are np arrays of spiketrains in timebin sized bins
        __get_unit_firing_rates__: Calculates  and returns firing rates per unit per recording in collection,
                    creates a dictionary and assigns it as recording.unit_firing_rates
                    the keys are unit ids (int) and values are firing rates for the
                    unit (np array) in timebin sized bins calculated using smoothing_window for averaging
        __get_event_snippets__: returns snippets of spiketrains or firing rates for events of the same
                    length, optional pre-event and post-event windows (s)
        __get_unit_event_firing_rates__: Calculates and returns firing rates for events per unit
        __wilcox_baseline_v_event_stats__: calculates wilcoxon signed-rank test for average firing rates
                    of two windows: event vs baseline where baseline is an amount of time immediately
                    prior to the event. Creates a dataframe of wilcoxon stats and p values for every unit.
                    Save for export optional.
        __wilcox_baseline_v_event_plots__: plots event triggered average firing rates for units with significant wilcoxon
                    signed rank tests (p value < 0.05) for event vs base line window.
        __wilcox_event_v_event_stats__: calculates wilcoxon signed-rank test for average firing rates between
                    two events for a given recording. Returns dataframe of wilcoxon stats
                    and p values for every unit is added to a dictionary of dataframes for that
                    recording. Key for this dictionary item is '{event1 } vs {event2} ({event_length}s)'
                    and the value is the dataframe. Option to save as attribute for the recording.
        __wilcox_event_v_event_plots__: plots event triggered average firing rates for units with significant wilcoxon
                    signed rank sums (p value < 0.05) for event1 vs event2
        __zscore_event__: Calculates zscored event average firing rates per unit including a baseline window (s).
                    Takes in a recording and an event and returns a dictionary of unit ids to z scored
                    averaged firing rates. It also assigns this dictionary as the value to a zscored event
                    dictionary of the recording such that the key is {event_length}s {event} vs {baseline_window}s baseline'
                    and the value is {unit id: np.array(zscored average event firing rates)}
        __zscore_plot__: plots z-scored average event firing rate for the population of good units with SEM
                    and the z-scored average event firing rate for each good unit individually for each recording
                    in the collection.
        __PCA_EDA_plot__: plots the first 2 PCs from the  PCA trajectories calculated in the last run of
                    PCA_trajectories with beginning of baseline, event onset, event end,
                    and end of post_window noted in graph

    """

    def __init__(self, ephyscollection, timebin, ignore_freq=0.1, smoothing_window=None, mode="same"):
        self.ephyscollection = ephyscollection
        self.timebin = timebin
        self.ignore_freq = ignore_freq
        self.smoothing_window = smoothing_window
        self.mode = mode
        self.PCA_matrix = None
        self.__all_set__()

    def __all_set__(self):
        """
        double checks that all SpikeRecordings in the collection have attributes: subject & event_dict and that
        each event_dict has the same keys. Warns users which recordings are missing subjects or event_dicts.
        If all set, prints "All set to analyze" and calculates spiketrains and firing rates.
        """
        is_first = True
        is_good = True
        missing_events = []
        missing_subject = []
        event_dicts_same = True
        event_type = False
        for recording_name, recording in self.ephyscollection.collection.items():
            if not hasattr(recording, "event_dict"):
                missing_events.append(recording_name)
            else:
                if is_first:
                    last_recording_events = recording.event_dict.keys()
                    is_first = False
                else:
                    if recording.event_dict.keys() != last_recording_events:
                        event_dicts_same = False
                for value in recording.event_dict.values():
                    if type(value) is np.ndarray:
                        if (value.ndim == 2) & (value.shape[1] == 2):
                            event_type = True
            if not hasattr(recording, "subject"):
                missing_subject.append(recording_name)
        if len(missing_events) > 0:
            print("These recordings are missing event dictionaries:")
            print(f"{missing_events}")
            is_good = False
        else:
            if not event_dicts_same:
                print("Your event dictionary keys are different across recordings.")
                print("Please double check them:")
                for (
                    recording_name,
                    recording,
                ) in self.ephyscollection.collection.items():
                    print(recording_name, "keys:", recording.event_dict.keys())
                is_good = False
        if len(missing_subject) > 0:
            print(f"These recordings are missing subjects: {missing_subject}")
            is_good = False
        if not event_type:
            print("Event arrays are not 2 dimensional numpy arrays of shape (n x 2).")
            print("Please fix.")
        if is_good:
            print("All set to analyze")
            self.__create_freq_dictionary__()
            self.__get_whole_spiketrain__()
            self.__get_unit_spiketrains__()
            self.__get_unit_firing_rates__()

    def __create_freq_dictionary__(self):
        sampling_rate = self.ephyscollection.sampling_rate
        for recording in self.ephyscollection.collection.values():
            last_timestamp = recording.timestamps_var[-1]
            freq_dict = {}
            for unit in recording.unit_timestamps.keys():
                if recording.labels_dict[str(unit)] == "good":
                    no_spikes = len(recording.unit_timestamps[unit])
                    unit_freq = no_spikes / last_timestamp * sampling_rate
                    freq_dict[unit] = unit_freq
            recording.freq_dict = freq_dict

    def __get_whole_spiketrain__(self):
        """
        creates a spiketrain for each recording where each array element is the number of spikes per timebin
        and assigns as .spiketrain for each recording
        """
        for recording in self.ephyscollection.collection.values():
            last_timestamp = recording.timestamps_var[-1]
            recording.spiketrain = fr.get_spiketrain(
                recording.timestamps_var, last_timestamp, self.timebin, recording.sampling_rate
            )

    def __get_unit_spiketrains__(self):
        """
        Creates a dictionary and assigns it as recording.unit_spiketrains for each recording.
        Only 'good' unit ids (not 'mua') with firing rates > ignore_freq are included.
        Keys are unit ids (ints) and values are numpy arrays of spiketrains in timebin-sized bins
        """
        sampling_rate = self.ephyscollection.sampling_rate
        for name, recording in self.ephyscollection.collection.items():
            last_timestamp = recording.timestamps_var[-1]
            unit_spiketrains = {}
            for unit in recording.freq_dict.keys():
                if recording.freq_dict[unit] > self.ignore_freq:
                    unit_spiketrains[unit] = fr.get_spiketrain(
                        recording.unit_timestamps[unit],
                        last_timestamp,
                        self.timebin,
                        sampling_rate,
                    )
                recording.unit_spiketrains = unit_spiketrains

    def __get_unit_firing_rates__(self):
        """
        Calculates firing rates per unit, creates a dictionary and assigns it as recording.unit_firing_rates
        Keys are unit ids (int) and values are numpy arrays of firing rates (Hz) in timebin sized bins
        Calculated using smoothing_window for averaging
        Creates a multi dimensional array as recording.unit_firing_rate_array of timebins x units
        """
        for recording in self.ephyscollection.collection.values():
            unit_firing_rates = {}
            for unit in recording.unit_spiketrains.keys():
                unit_firing_rates[unit] = fr.get_firing_rate(
                    recording.unit_spiketrains[unit], self.timebin, self.smoothing_window, self.mode
                )
            recording.unit_firing_rates = unit_firing_rates
            recording.unit_firing_rate_array = np.array([unit_firing_rates[key] for key in unit_firing_rates]).T

    def __get_event_snippets__(self, recording, event, whole_recording, event_length, pre_window=0, post_window=0):
        """
        takes snippets of spiketrains or firing rates for events with optional pre-event and post-event windows (s)
        all events must be of equal length (extends snippet lengths for events shorter then event_length and trims those
        that are longer)

        Args (6 total, 4 required):
            recording: SpikeRecording instance, recording to get snippets
            event: str, event type of which ephys snippets happen during
            whole_recording: numpy array, spiketrain or firing rates for the whole recording
            event_length: float, length (s) of events used through padding and trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Returns (1):
            event_snippets: a list of lists, where each list is a list of
                firing rates or spiketrains during an event including
                pre_window & post_windows, accounting for event_length and
                timebins for a single unit or for the population returning
                a list of numpy arrays
        """
        if type(event) is str:
            events = recording.event_dict[event]
        else:
            events = event
        event_snippets = []
        pre_window = round(pre_window * 1000)
        post_window = round(post_window * 1000)
        event_length = event_length * 1000
        event_len = int((event_length + pre_window + post_window) / self.timebin)
        print("event len", event_len)
        for i in range(events.shape[0]):
            pre_event = int((events[i][0] - pre_window) / self.timebin)
            post_event = pre_event + event_len
            if len(whole_recording.shape) == 1:
                event_snippet = whole_recording[pre_event:post_event]
                print("single unit start and stop", pre_event, post_event)
                # drop events that start before the beginning of the recording
                # given a long prewindow
                if pre_event >= 0:
                    # drop events that go beyond the end of the recording
                    if post_event < whole_recording.shape[0]:

                        event_snippets.append(event_snippet)
            else:
                event_snippet = whole_recording[pre_event:post_event, ...]
                print("recording array start and stop", pre_event, post_event)
                # drop events that start before the beginning of the recording
                # given a long prewindow
                if pre_event >= 0:
                    # drop events that go beyond the end of the recording
                    if post_event < whole_recording.shape[0]:
                        event_snippets.append(event_snippet)

        print(np.array(event_snippets).shape)
        # event_snippets = [trial, timebins, units] or [trial, timebin] per unit
        return event_snippets

    def __get_unit_event_firing_rates__(self, recording, event, event_length, pre_window=0, post_window=0):
        """
        returns firing rates for events per unit

        Args (5 total, 3 required):
            recording: SpikeRecording instance, recording for firing rates
            event: str, event type of which ehpys snippets happen during
            event_length: float, length (s) of events used by padding or trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Return (1):
            unit_event_firing_rates: dict, keys are unit ids (???),
            values are lsts of numpy arrays of firing rates per event
        """
        unit_event_firing_rates = {}
        for unit in recording.unit_firing_rates.keys():
            unit_event_firing_rates[unit] = self.__get_event_snippets__(
                recording,
                event,
                recording.unit_firing_rates[unit],
                event_length,
                pre_window,
                post_window,
            )
        return unit_event_firing_rates

    def __get_event_firing_rates__(self, recording, event, event_length, pre_window=0, post_window=0):
        """
        Grabs event firing rates from a whole recording through the recordings
        unit firing rate array (units by time bins)

        Args (5 total, 3 required):
            recording: SpikeRecording instance, recording for firing rates
            event: str, event type of which ehpys snippets happen during
            event_length: float, length (s) of events used by padding or trimming events
            pre_window: int, default=0, seconds prior to start of event
            post_window: int, default=0, seconds after end of event

        Returns (1):
            event_firing_rates: list of arrays, where each array
            is timebins x units and list is len(no of events)
        """
        event_firing_rates = self.__get_event_snippets__(
            recording, event, recording.unit_firing_rate_array, event_length, pre_window, post_window
        )
        return event_firing_rates

    def __wilcox_baseline_v_event_stats__(
        self, recording, event, event_length, baseline_window, offset, exclude_offset
    ):
        preevent_baselines = np.array(
            [wilcox.pre_event_window(event, baseline_window, offset) for event in recording.event_dict[event]]
        )
        if len(recording.event_dict[event]) < 6:
            print(f"Wilcoxon can't be done on {recording.name} {event}, because <6 samples")
            return pd.DataFrame({"Wilcoxon Stat": [np.nan], "p value": [np.nan]})
        unit_baseline_firing_rates = self.__get_unit_event_firing_rates__(
            recording, preevent_baselines, event_length=(baseline_window + offset), pre_window=0, post_window=0
        )
        if exclude_offset:
            unit_event_firing_rates = self.__get_unit_event_firing_rates__(recording, event, event_length, 0, 0)
        else:
            unit_event_firing_rates = self.__get_unit_event_firing_rates__(recording, event, event_length, -(offset), 0)
        try:
            unit_averages = wilcox.event_avgs(unit_event_firing_rates, unit_baseline_firing_rates)
            # for unit in unit_averages.keys():
            #     if unit_averages[unit][0] == unit_averages[unit][1]:
            #         print(f"Wilcoxon can't be done on {recording.name} {event} {unit}, because baseline = event")
            #         unit_averages[unit] = [np.nan, np.nan]  # Set the unit's values to NaN
        except Exception as e:
            print(f"{recording.name}")
        wilcox_df = wilcox.signed_rank(unit_averages)
        return wilcox_df

    def wilcox_baseline_v_event(
        self, event, event_length, baseline_window, offset=0, exclude_offset=False, plot=True, save=False
    ):
        """
        Runs a wilcoxon signed rank test on all good units from all recordings on the
        given event's firing rate versus the given baseline window immediately prior to event onset.

        Creates a dataframe with rows for each unit and columns representing Wilcoxon stats, p values, orginal unit ids, recording,
        subject and the event + baselien given.

        Args(4 total, 3 required):
            event: str, event firing rates for stats to be run on
            event_length: float, length (s) of events used by padding or trimming events all to event_length (s)
            baseline_window: int, default=0, seconds prior to start of event
           offset: int, adjusts end of baseline by offset(s) from onset of behavior i.e. offset=2 adds the first two seconds of event
                data into baseline while offest=-2 removes them from baseline averages
            exclude_offset: Boolean, default=False, if true excludes time prior to onset and before offset in event averages, if false,
                time between onset and offset are included in event averages
            plot: Boolean, default=True, if True, plots, if False, does not plot.

        Returns(1):
            master_df: df, rows for each unit and columns representing
                Wilcoxon stats, p values, orginal unit ids, recording
        """
        is_first = True
        for recording_name, recording in self.ephyscollection.collection.items():
            recording_df = self.__wilcox_baseline_v_event_stats__(
                recording, event, event_length, baseline_window, offset, exclude_offset, save
            )
            recording_df = recording_df.reset_index().rename(columns={"index": "original unit id"})
            recording_df["Recording"] = recording_name
            recording_df["Subject"] = recording.subject
            recording_df["Event"] = f"{event_length}s {event} vs {baseline_window}s baseline"
            if is_first:
                master_df = recording_df
                is_first = False
            else:
                master_df = pd.concat([master_df, recording_df], axis=0).reset_index(drop=True)
        if plot:
            for name, recording in self.ephyscollection.collection.items():
                unit_event_firing_rates = self.__get_unit_event_firing_rates__(
                    recording, event, event_length, baseline_window
                )
                wilcox.baseline_v_event_plot(
                    name, unit_event_firing_rates, master_df, event, event_length, baseline_window, offset
                )
        return master_df

    def __wilcoxon_event_v_event_stats__(self, recording, event1, event2, event_length):
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
        unit_event1_firing_rates = self.__get_unit_event_firing_rates__(recording, event1, event_length, 0, 0)
        unit_event2_firing_rates = self.__get_unit_event_firing_rates__(recording, event2, event_length, 0, 0)
        if (len(recording.event_dict[event1]) < 6) | (len(recording.event_dict[event2]) < 6):
            print(f"Wilcoxon can't be done on {recording.name} because <6 samples for either {event1} or {event2}")
            return pd.DataFrame({"Wilcoxon Stat": [np.nan], "p value": [np.nan]})
        try:
            unit_averages = wilcox.event_avgs(unit_event1_firing_rates, unit_event2_firing_rates)
            for unit in unit_averages.keys():
                if unit_averages[unit][0] == unit_averages[unit][1]:
                    print(
                        f"Wilcoxon can't be done on {recording.name} {event1} and {event2} {unit}, because baseline = event"
                    )
                    unit_averages[unit] = [np.nan, np.nan]  # Set the unit's values to NaN
        except Exception as e:
            print(f"{recording.name}")
        wilcoxon_df = wilcox.rank_sum(unit_averages)
        return wilcoxon_df

    def wilcox_event_v_event(self, event1, event2, event_length, pre_window, plot=True):
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
        for (
            recording_name,
            recording,
        ) in self.ephyscollection.collection.items():
            recording_df = self.__wilcoxon_event_v_event_stats__(recording, event1, event2, event_length)
            recording_df = recording_df.reset_index().rename(columns={"index": "original unit id"})
            recording_df["Recording"] = recording_name
            recording_df["Subject"] = recording.subject
            recording_df["Event"] = f"{event1 } vs {event2} ({event_length}s)"
            if is_first:
                master_df = recording_df
                is_first = False
            else:
                master_df = pd.concat([master_df, recording_df], axis=0).reset_index(drop=True)
        if plot:
            for name, recording in self.ephyscollection.collection.items():
                unit_event1_firing_rates = self.__get_unit_event_firing_rates__(
                    recording, event1, event_length, pre_window
                )
                unit_event2_firing_rates = self.__get_unit_event_firing_rates__(
                    recording, event2, event_length, pre_window
                )
                wilcox.event_v_event_plot(
                    name, unit_event1_firing_rates, unit_event2_firing_rates, master_df, event_length, pre_window
                )
        return master_df

    def fisher_exact_wilcox(
        self, event1, event2, event_length, event3=None, baseline_window=None, offset=0, exclude_offset=False
    ):
        """
        Calculates fisher's exact test: contigency matrix is made up of number of significant units from wilcoxon signed rank test of
        baseline_window vs event vs non-significant units for event1 and event2.

        Args(7 total, 3 required):
            event1: str, event in event_dict
            event2: str, event in event_dict
            event3: str, default=None, if not none, event type to be used as baseline in wilcoxon signed rank sum test
            event_length: int, length (s) of event to be used
            baselin_window: int, default=None, if not None, length (s) of baseline window immediately prior to event onset
            offset: int, adjusts end of baseline by offset(s) from onset of behavior i.e. offset=2 adds the first two seconds of event
                data into baseline while offest=-2 removes them from baseline averages
            exclude_offset: Boolean, default=False, if true excludes time prior to onset and before offset in event averages, if false,
                time between onset and offset are included in event averages

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
            df1 = self.wilcox_baseline_v_event(
                event1, event_length, baseline_window, offset, exclude_offset, plot=False
            )
            df2 = self.wilcox_baseline_v_event(
                event2, event_length, baseline_window, offset, exclude_offset, plot=False
            )
        else:
            df1 = self.wilcox_event_v_event(event1, event3, event_length, plot=False)
            df2 = self.wilcox_event_v_event(event2, event3, event_length, plot=False)
        row1 = [(df1["p value"] < 0.05).sum(), (df1["p value"] > 0.05).sum()]
        row2 = [(df2["p value"] < 0.05).sum(), (df2["p value"] > 0.05).sum()]
        contingency_matrix = [row1, row2]
        odds_ratio, p_value = fisher_exact(contingency_matrix)
        return odds_ratio, p_value, contingency_matrix
