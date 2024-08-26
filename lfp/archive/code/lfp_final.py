import pandas as pd
import os
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

class LFPRecording:
    #could change ecu & trodes to boolean values
    def __init__(self,
                 path,
                 channel_map_path,
                 events_path,
                 subject,
                 ecu=False,
                 sampling_rate=20000,
                 ecu_stream_id="ECU",
                 trodes_stream_id="trodes",
                 lfp_freq_min=0.5,
                 lfp_freq_max=300,
                 electric_noise_freq=60,
                 lfp_sampling_rate=1000,
                 frame_rate=22):

        self.path = path
        self.channel_map_path = channel_map_path
        self.events_path = events_path

        self.events = {}
        self.channel_map = {}
        self.recording = None

        self.subject = subject

        self.sampling_rate = sampling_rate
        self.ecu_stream_id = ecu_stream_id
        self.trodes_stream_id = trodes_stream_id
        self.lfp_freq_min = lfp_freq_min
        self.lfp_freq_max = lfp_freq_max
        self.electric_noise_freq = electric_noise_freq
        self.lfp_sampling_rate = lfp_sampling_rate
        self.frame_rate = frame_rate

        self.make_recording()
        self.make_events()
        self.make_channel_map()
        self.get_lfp_traces()
        self.ecu = ecu

        #making a function to calculate power at each frequency


        print(self.recording)
        print(self.events)
        print(self.channel_map.columns)
        print(self.subject)

        # input needs to be zeroed to stream
        # spike gadgets takes time start & stop but output is indexed on the index column

        # time vs time_stamp_index --> spike gadgets takes time start & stop but output is indexed on the index column
        # everything in output is shifted (by time_stamp_index)
        # off_setting start stop when you hit record
        # offset is in merge.rec folder --> avoid user calculation
            # zero on recording
        # lfp indexed to stream

        # if ecu: zeroed on stream
        # if not ecu: it is zeroed on recording
            # MINUS offset on start and stop times

        # offset is ONLY needed for ECU data
        # trial index and add 10,000 and every spike index that falls in that range is used

    def make_events(self):
        # read channel map
        # read events
        temp_events_df = pd.read_excel(self.events_path)
        # lower case all column names
        temp_events_df.columns = map(str.lower, temp_events_df.columns)
        # choose only required columns --> event, subject, time_start, time_stop
        temp_events_df = temp_events_df[["event", "subject", "time_start", "time_stop"]]
        #convert subject to string
        temp_events_df["subject"] = temp_events_df["subject"].astype(str)

        # choose only rows with current subject
        temp_events_df = temp_events_df[temp_events_df["subject"] == self.subject]

        for index, row in temp_events_df.iterrows():
            #use tuple to store start and stop times
            if row["event"] not in self.events:
                self.events[row["event"]] = []
            self.events[row["event"]].append((row["time_start"], row["time_stop"]))

    def make_recording(self):
            # change to try except, check for corrupted data and continue
            # look into making a new variable
            # calculate events from ecu data
        current_recording = se.read_spikegadgets(self.path, stream_id=self.ecu_stream_id)
        current_recording = se.read_spikegadgets(self.path, stream_id=self.trodes_stream_id)
        current_recording = sp.bandpass_filter(current_recording, freq_min=self.lfp_freq_min, freq_max=self.lfp_freq_max)
        current_recording = sp.notch_filter(current_recording, freq=self.electric_noise_freq)
        current_recording = sp.resample(current_recording, resample_rate=self.lfp_sampling_rate)
        current_recording = sp.zscore(current_recording)
        self.recording = current_recording

    def make_channel_map(self):
        # only get info for current subject
        channel_map_df = pd.read_excel(self.channel_map_path)
        # lowercase all column names
        channel_map_df.columns = map(str.lower, channel_map_df.columns)
        channel_map_df["subject"] = channel_map_df["subject"].astype(str)

        channel_map_df = channel_map_df[channel_map_df["subject"] == self.subject]
        self.channel_map = channel_map_df



    def get_lfp_traces(self):
        brain_region = []
        for col in self.channel_map.columns:
            if "spike_interface" in col:
                brain_region.append(col)
        print(brain_region)
        for col in brain_region:
            self.channel_map[col] = self.channel_map[col].astype(int).astype(str)
            """
            channel_map_and_all_trials_df["{}_baseline_lfp_trace".format(col.strip("spike_interface").strip("_"))] = channel_map_and_all_trials_df.apply(lambda row: row["all_ch_lfp"].get_traces(channel_ids=[row[col]], start_frame=row["baseline_lfp_timestamp_range"][0], end_frame=row["baseline_lfp_timestamp_range"][1]).T[0], axis=1)

            channel_map_and_all_trials_df["{}_trial_lfp_trace".format(col.strip("spike_interface").strip("_"))] = channel_map_and_all_trials_df.apply(lambda row: row["all_ch_lfp"].get_traces(channel_ids=[row[col]], start_frame=row["trial_lfp_timestamp_range"][0], end_frame=row["trial_lfp_timestamp_range"][1]).T[0], axis=1)
            """
            # get from spike recording
            self.channel_map[
                "{}_baseline_lfp_trace".format(col.strip("spike_interface").strip("_"))] = self.channel_map.apply(
                lambda row:
                self.recording.get_traces(channel_ids=[row[col]], start_frame=row["baseline_lfp_timestamp_range"][0],
                                          end_frame=row["baseline_lfp_timestamp_range"][1]).T[0], axis=1)
            self.channel_map[
                "{}_trial_lfp_trace".format(col.strip("spike_interface").strip("_"))] = self.channel_map.apply(
                lambda row:
                self.recording.get_traces(channel_ids=[row[col]], start_frame=row["trial_lfp_timestamp_range"][0],
                                          end_frame=row["trial_lfp_timestamp_range"][1]).T[0], axis=1)
        print(self.channel_map)


class LFPrecordingCollection:
    def __init__(self, path, channel_map_path, events_path, sampling_rate=1000):
        self.path = path
        self.channel_map_path = channel_map_path
        self.sampling_rate = sampling_rate
        self.events_path = events_path
        self.collection = {}
        self.make_collection()

        # boolean ecu true or false
            # if ecu: scrapped & calculated from recording (rather than excel)
    def make_collection(self):
        collection = {}
        for root, dirs, files in os.walk(self.path):
            for directory in dirs:
                if directory.endswith("merged.rec"):
                    for file in os.listdir(os.path.join(root, directory)):
                        #handle channel map before recording
                        collection[directory] = {}
                        recording_path = os.path.join(root, directory, file)
                        subject = "1.4" #TODO: TEMP FIX FOR SUBJECT
                        #TODO: user input for subject name per recording in list
                        #do not extract name from recording or others
                        recording = LFPRecording(recording_path, self.channel_map_path, self.events_path, subject)
                        #add to collection at subject
                        collection[directory][subject] = recording
        self.collection = collection

testData = LFPrecordingCollection("reward_competition_extention/data/omission/test/","channel_mapping.xlsx", "test.xlsx")