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
        self.ecu = ecu

        print(self.recording)
        print(self.events)
        print(self.channel_map)

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
        print(temp_events_df.columns)
        # lower case all column names
        temp_events_df.columns = map(str.lower, temp_events_df.columns)
        # choose only required columns --> event, subject, time_start, time_stop
        temp_events_df = temp_events_df[["event", "subject", "time_start", "time_stop"]]
        # convert to dictionary with key as subject name and value as dictionary of events
        # dictionary of events = key as event name and value as list of times

        temp_events_df = temp_events_df.set_index("subject")
        for subject in temp_events_df.index:
            self.events[subject] = {}
            #only check for current subject
            if subject != self.subject:
                continue
            else:
                for event in temp_events_df.loc[subject]["event"]:
                    self.events[subject][event] = []
                for event, time_start, time_stop in zip(temp_events_df.loc[subject]["event"],
                                                        temp_events_df.loc[subject]["time_start"],
                                                        temp_events_df.loc[subject]["time_stop"]):
                    self.events[subject][event].append((time_start, time_stop))

    def make_recording(self):
        if self.ecu:
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

        channel_map_df = channel_map_df[channel_map_df["subject"] == self.subject]
        self.channel_map = channel_map_df.to_dict()


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