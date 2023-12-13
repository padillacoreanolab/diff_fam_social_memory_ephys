import pandas as pd
import os
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp

class LFPRecording:
    #could change ecu & trodes to boolean values
    def __init__(self,
                 path,
                 channel_map_path,
                 sampling_rate=20000,
                 ecu_stream_id="ECU",
                 trodes_stream_id="trodes",
                 lfp_freq_min=0.5,
                 lfp_freq_max=300,
                 electric_noise_freq=60,
                 lfp_sampling_rate=1000,
                 frame_rate=22):
        self.path = path
        self.sampling_rate = sampling_rate
        self.events = {} #create from channel map df
        self.channel_map_df = pd.read_excel(channel_map_path)
        self.ecu_stream_id = ecu_stream_id
        self.trodes_stream_id = trodes_stream_id
        self.lfp_freq_min = lfp_freq_min
        self.lfp_freq_max = lfp_freq_max
        self.electric_noise_freq = electric_noise_freq
        self.lfp_sampling_rate = lfp_sampling_rate
        self.frame_rate = frame_rate
        def channel_map(path):
            return []

        for file in os.listdir(self.path):
            print("file is " + file)
            if file.endswith(".rec"):
                print("file ends with .rec")
                print("reading for recording name " + file)
                absolute_path = os.path.join(self.path, file)
                current_recording = se.read_spikegadgets(absolute_path, stream_id=ecu_stream_id)
                current_recording = se.read_spikegadgets(absolute_path, stream_id=trodes_stream_id)
                current_recording = sp.bandpass_filter(current_recording, freq_min=lfp_freq_min, freq_max=lfp_freq_max)
                current_recording = sp.notch_filter(current_recording, freq=electric_noise_freq)
                current_recording = sp.resample(current_recording, resample_rate=lfp_sampling_rate)
                current_recording = sp.zscore(current_recording)
                print("after z score")
                print(current_recording)



        #read channel map
        #create events dictionary


class LFPrecordingCollection:
    def __init__(self, path, channel_map_path, sampling_rate=1000):
        self.path = path
        self.sampling_rate = sampling_rate

        self.channel_map_df = pd.read_excel(channel_map_path)
        self.make_collection()

    def make_collection(self):
        collection = {}
        data = pd.read_excel(self.path)
        for root, dirs, files in os.walk(self.path):
            for directory in dirs:
                if directory.endswith("merged.rec"):
                    for file in os.listdir(os.path.join(root, directory)):
                        #handle channel map before recording
                        collection[directory] = {}
                        recording = LFPRecording(os.path.join(root, directory, file))
                        collection[directory]["recording"] = recording
                        collection[directory]["channel_map"] = LFPRecording
        self.collection = collection

testData = LFPRecording("reward_competition_extention/data/omission/test/test_1_merged.rec", "test.xlsx")