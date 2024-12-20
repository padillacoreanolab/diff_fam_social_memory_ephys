def __global_baseline__(self, recording, event, equalize, pre_window, global_timebin):
        """
        calculates baseline firing rate dicitionaries of size global_timebin across entire recording
        for each unit and creates a unique caching name for zscore_dict for the collection
        and the recording

        Args (5 total):
            recordng: ephysrecording instance
            event: str, event type
            equalize: int, length (s) of event
            pre_window: int, time (s) before onset of event
            global_timebin: length (ms) of timebins to calculate mew and sigma for zscore

        Returns (2):
            unit_baseline_firing_rates: dict; key, str, unit id
                value: numpy array, 2d numpy array of global time bin chunks of firing rates
            event_name: str, event name for caching
        """
        unit_firing_rates = recording.unit_firing_rates
        unit_baseline_firing_rates = {
            key: chunk_array(value, global_timebin, self.timebin) for key, value in unit_firing_rates.items()
        }
        event_name = f"{equalize}s {event} w/ pre {pre_window}s vs global ({global_timebin})"
        return unit_baseline_firing_rates, event_name

    def __event_baseline__(self, recording, event, baseline, equalize, pre_window):
        """
        calculates baseline firing rate dicitionaries of baseline events
        for each unit and creates a unique caching name for zscore_dict for
        the collection and the recording

        Args (5 total):
            recordng: ephysrecording instance
            event: str, event type
            baseline: str, event type to be used as baseline for z score
            equalize: int, length (s) of event
            pre_window: int, time (s) before onset of event

        Returns (2):
            unit_baseline_firing_rates: dict; key, str, unit id
                value: numpy array, 2d numpy array of baseline event firing rates
            event_name: str, event name for caching
        """
        unit_baseline_firing_rates = self.__get_unit_event_firing_rates__(recording, baseline, equalize, pre_window)
        event_name = f"{equalize}s {event} vs {baseline} baseline (w/ pre {pre_window}s)"
        return unit_baseline_firing_rates, event_name

    def __calc_preevent_baseline__(self, recording, baseline, offset, equalize, event):
        """
        calculates baseline firing rate dicitionaries of pre-event baseline windows
        for each unit and creates a unique caching name for zscore_dict for
        the collection and the recording

        Args (4 total):
            recordng: ephysrecording instance
            event: str, event type
            baseline: int, time (s) before onset of event
            equalize: int, length (s) of event

        Returns (2):
            unit_baseline_firing_rates: dict; key, str, unit id
                value: numpy array, 2d numpy array of baseline pre-event firing rates
            event_name: str, event name for caching
        """
        preevent_baselines = np.array(
            [pre_event_window(event, baseline, offset) for event in recording.event_dict[event]]
        )
        unit_baseline_firing_rates = self.__get_unit_event_firing_rates__(recording, preevent_baselines, baseline, 0, 0)
        event_name = f"{equalize}s {event} vs {baseline}s baseline"
        return unit_baseline_firing_rates, event_name

    def __zscore_event__(self, recording, unit_event_firing_rates, unit_baseline_firing_rates, SD=None):
        """
        Calculates zscored event average firing rates per unit including a baseline window (s).
        Takes in a recording and an event and returns a dictionary of unit ids to z scored
        averaged firing rates.
        It also assigns this dictionary as the value to a zscored event dictionary of the recording.
        Such that the key is {equalize}s {event} vs {baseline_window}s baseline'
        and the value is {unit id: np.array(zscored average event firing rates)}

        Args(4 total, 3 required):
            recording: EphysRecording instance, recording that is being zscored
            unit_event_firing_rates: numpy array, event firing rates to be z scored
            unit_baseline_firing_rates: numpy array, baseling firing rates to be z scored to
                whic mew and sigma will be calculated from
            SD: int, deault=None, number of standard deviations away from the mean for a global
                zscored event to be considered significant

        Returns(1, 2 optional):
            zscored_events: dict, of units to z scored average event firing rates
                keys: str, unit ids
                values: np.array, average z scared firing rates
            significance_dict: dict, if SD is not none, units to significance
                keys: str, unit ids,
                values: 'inhibitory' if average zscored event is under SD * sigma ;
                    'excitatory' if average zscored event firing is over SD * sigma
                    'not significant' if none of the above
        """
        zscored_events = {}
        significance_dict = {}
        for unit in unit_event_firing_rates:
            # calculate average event across all events per unit
            event_average = np.mean(unit_event_firing_rates[unit], axis=0)
            # one average for all preevents
            baseline_average = np.mean(unit_baseline_firing_rates[unit], axis=0)
            mew = np.mean(baseline_average)
            sigma = np.std(baseline_average)
            if sigma != 0:
                zscored_event = [(event_bin - mew) / sigma for event_bin in event_average]
                if SD is not None:
                    significance = ""
                    if np.mean(zscored_event) < -(SD * sigma):
                        significance = "inhibitory"
                    if np.mean(zscored_event) > SD * sigma:
                        if significance == "inhibitory":
                            significance = "both?"
                        else:
                            significance = "excitatory"
                    else:
                        significance = "not significant"
                    significance_dict[unit] = significance
                zscored_events[unit] = zscored_event
        if SD is not None:
            return zscored_events, significance_dict
        else:
            return zscored_events

    def __make_zscore_df__(self, zscored_events, recording, recording_name, event_name, master_df=None, sig_dict=None):
        """
        Args(4 required, 6 total):
            zscored_events: dict, unit ids as keys, z scored firing rates of values (numpy arrays)
            recording: ephys recording instance
            recording_name: str, name of the recording
            event_name: str, cached event name with parameters
            master_df: dataframe, curent z scored df, none if first event calculated
            sig_dict: dict, significance of each unit based on event average compared to sigma

        Returns:
            master_df: dataframe containing z scored events, subject, event name, recording name,
                and optional significance status (all columns) per unit (rows)

        """
        zscored_events_df = pd.DataFrame.from_dict(zscored_events, orient="index")
        if sig_dict is not None:
            zscored_events_df.insert(0, "Significance", [sig_dict[i] for i in zscored_events_df.index])
        zscored_events_df = zscored_events_df.reset_index().rename(columns={"index": "original unit id"})
        zscored_events_df.insert(0, "Subject", recording.subject)
        zscored_events_df.insert(0, "Event", event_name)
        zscored_events_df.insert(0, "Recording", recording_name)
        if master_df is None:
            master_df = zscored_events_df
        else:
            master_df = pd.concat([master_df, zscored_events_df], axis=0).reset_index(drop=True)
        return master_df

    def zscore_global(self, event, equalize, pre_window=0, global_timebin=1000, SD=None, plot=True, save=False):
        """
        calculates z-scored event average firing rates for all recordings in the collection
        compared to a global baseline firing rate in global_timebin (ms) chunks
        assigns a dataframe of all zscored event firing rates with columns for original unit id,
        recording name, and subject as a value in zscored_event dictionary attribute of the colleciton

        Args (7 total, 2 required):
            event: str, event type whose average firing rates are being z-scored
            equalize: float, length (s) of events used by padding with post event time
                    or trimming events all to equalize (s) long used in z scoring
            pre_window: float, default=0, firing rates prior to event to be zscored.
            global_timebin: float, default=1000, timebin (ms) with which mew and sigma will be calculated
                    for 'global' normalization against whole recording
            SD: int, default=None, number of standard of deviations away from the mean of the global z score
                it takes for a units average firing rate per event to be considered signifcant
            plot: Boolean, default=True, if true, function will plot, if false, function will not plot
                    z scored event over time (pre_window to equalize)
            save: Boolean, default=False, if False, will not cache results for export, if True, will
                  will save results in collection.zscored_events dict for export

        Returns:
            master_df: assigns a dataframe of all zscored event firing rates with columns for original unit id,
                   recording name, and subject as a value in zscored_event dictionary attribute of the colleciton
                   such that: collection the key is '{event} vs {baseline_window}s baseline' and the value is the
                   dataframe
        """
        is_first = True
        zscored_dict = {}
        for recording_name, recording in self.ephyscollection.collection.items():
            unit_event_firing_rates = self.__get_unit_event_firing_rates__(recording, event, equalize, pre_window, 0)
            unit_baseline_firing_rates, event_name = self.__global_baseline__(
                recording, event, equalize, pre_window, global_timebin
            )
            zscored_events, significance_dict = self.__zscore_event__(
                recording, unit_event_firing_rates, unit_baseline_firing_rates, SD
            )
            if save:
                recording.zscored_events[event_name] = zscored_events
            zscored_dict[recording_name] = zscored_events
            if is_first:
                master_df = self.__make_zscore_df__(
                    zscored_events, recording, recording_name, event_name, master_df=None, sig_dict=significance_dict
                )
                is_first = False
            else:
                master_df = self.__make_zscore_df__(
                    zscored_events, recording, recording_name, event_name, master_df, sig_dict=significance_dict
                )
        if save:
            self.ephyscollection.zscored_events[event_name] = master_df
        if plot:
            self.__zscore_plot__(zscored_dict, event, equalize, pre_window)
        return master_df

    def zscore_baseline_event(self, event, baseline, equalize, pre_window=0, plot=True, save=False):
        """
        calculates z-scored event average firing rates for all recordings in the collection
        compared to a baseline event
        assigns a dataframe of all zscored event firing rates with columns for original unit id,
        recording name, and subject as a value in zscored_event dictionary attribute of the colleciton

        Args (6 total, 3 required):
            event: str, event type whose average firing rates are being z-scored
            baseline: str, baseline event to which other events will be normalized to
            equalize: float, length (s) of events used by padding with post event time
                    or trimming events all to equalize (s) long used in z scoring
            pre_window: float, default=0, firing rates prior to event to be normalized
            plot: Boolean, default=True, if true, function will plot, if false, function will not plot
                    z scored event over time (pre_window to equalize)
            save: Boolean, default=False, if False, will not cache results for export, if True, will
                  will save results in collection.zscored_events dict for export

        Returns:
            master_df: assigns a dataframe of all zscored event firing rates with columns for original unit id,
                   recording name, and subject as a value in zscored_event dictionary attribute of the colleciton
                   such that: collection the key is '{event} vs {baseline_window}s baseline' and the value is the
                   dataframe
        """
        is_first = True
        zscored_dict = {}
        for recording_name, recording in self.ephyscollection.collection.items():
            unit_event_firing_rates = self.__get_unit_event_firing_rates__(recording, event, equalize, pre_window, 0)
            unit_baseline_firing_rates, event_name = self.__event_baseline__(
                recording, event, baseline, equalize, pre_window
            )
            zscored_events = self.__zscore_event__(recording, unit_event_firing_rates, unit_baseline_firing_rates)
            if save:
                recording.zscored_events[event_name] = zscored_events
            zscored_dict[recording_name] = zscored_events
            if is_first:
                master_df = self.__make_zscore_df__(
                    zscored_events, recording, recording_name, event_name, master_df=None
                )
                is_first = False
            else:
                master_df = self.__make_zscore_df__(zscored_events, recording, recording_name, event_name, master_df)
        if save:
            self.ephyscollection.zscored_events[event_name] = master_df
        if plot:
            self.__zscore_plot__(zscored_dict, event, equalize, pre_window)

    def zscore_pre_event(self, event, equalize, baseline_window, offset=0, plot=True, save=False):
        """
        calculates z-scored event average firing rates for all recordings in the collection
        compared to a baseline window immediately prior to event onset.
        assigns a dataframe of all zscored event firing rates with columns for original unit id,
        recording name, and subject as a value in zscored_event dictionary attribute of the colleciton

        Args (5 total, 3 required):
            event: str, event type whose average firing rates are being z-scored
            baseline_window: str, baseline event to which other events will be normalized to
            equalize: float, length (s) of events used by padding with post event time
                    or trimming events all to equalize (s) long used in z scoring
            plot: Boolean, default=True, if true, function will plot, if false, function will not plot
                    z scored event over time (pre_window to equalize)
            save: Boolean, default=False, if False, will not cache results for export, if True, will
                  will save results in collection.zscored_events dict for export

        Returns:
            master_df: assigns a dataframe of all zscored event firing rates with columns for original unit id,
                   recording name, and subject as a value in zscored_event dictionary attribute of the colleciton
                   such that: collection the key is '{event} vs {baseline_window}s baseline' and the value is the
                   dataframe
        """
        is_first = True
        zscored_dict = {}
        for recording_name, recording in self.ephyscollection.collection.items():
            unit_event_firing_rates = self.__get_unit_event_firing_rates__(
                recording, event, equalize, baseline_window, 0
            )
            unit_baseline_firing_rates, event_name = self.__calc_preevent_baseline__(
                recording, baseline_window, offset, equalize, event
            )
            zscored_events = self.__zscore_event__(recording, unit_event_firing_rates, unit_baseline_firing_rates)
            if save:
                recording.zscored_events[event_name] = zscored_events
            zscored_dict[recording_name] = zscored_events
            if is_first:
                master_df = self.__make_zscore_df__(
                    zscored_events, recording, recording_name, event_name, master_df=None
                )
                is_first = False
            else:
                master_df = self.__make_zscore_df__(zscored_events, recording, recording_name, event_name, master_df)
        if save:
            self.ephyscollection.zscored_events[event_name] = master_df
        if plot:
            self.__zscore_plot__(zscored_dict, event, equalize, baseline_window, offset)
        return master_df

    def __zscore_plot__(self, zscored_dict, event, equalize, baseline_window, offset=0):
        """
        plots z-scored average event firing rate for the population of good units with SEM
        and the z-scored average event firing rate for each good unit individually for
        each recording in the collection.

        Args (4 total, 4 required):
            event: str, event type whose average z-scored firing rates will be plotted
            equalize: int, length (s) of event plotted
            baseline_window: int, length (s) of time prior to event onset plotted
            title: str, title of plot

        Return:
            none
        """
        no_plots = len(list(self.ephyscollection.collection.keys()))
        height_fig = no_plots
        i = 1
        plt.figure(figsize=(20, 4 * height_fig))
        for recording_name, recording in self.ephyscollection.collection.items():
            zscored_unit_event_firing_rates = zscored_dict[recording_name]
            zscore_pop = np.array(list(zscored_unit_event_firing_rates.values()))
            mean_arr = np.mean(zscore_pop, axis=0)
            sem_arr = sem(zscore_pop, axis=0)
            x = np.linspace(start=-baseline_window, stop=equalize, num=len(mean_arr))
            plt.subplot(height_fig, 2, i)
            plt.plot(x, mean_arr, c="b")
            plt.axvline(x=0, color="r", linestyle="--")
            if offset != 0:
                plt.axvline(x=offset, color="b", linestyle="--")
            plt.fill_between(x, mean_arr - sem_arr, mean_arr + sem_arr, alpha=0.2)
            plt.title(f"{recording_name} Population z-score")
            plt.subplot(height_fig, 2, i + 1)
            for unit in zscored_unit_event_firing_rates.keys():
                plt.plot(x, zscored_unit_event_firing_rates[unit], linewidth=0.5)
                plt.axvline(x=0, color="r", linestyle="--")
                plt.title(f"{recording_name} Unit z-score")
                if offset != 0:
                    plt.axvline(x=offset, color="b", linestyle="--")
            i += 2
        plt.suptitle(f"{equalize}s {event} vs {baseline_window}s baseline: Z-scored average")
        plt.show()