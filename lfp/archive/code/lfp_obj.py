import pandas as pd
import os

#assuming condition col = win/ loss --> unique --> all event types
#assuming subject_info is the subject attributed in the individual df

class LFPRecording:
    def __init__(self, path, sampling_rate=20000):
        self.path = path
        self.sampling_rate = sampling_rate
        self.events = {} #TONE TIME STAMP DF

        def __all_set__():
            self.events = set(self.events)
            #missing a channel map, check for subject
            #brain region : channel number
            #subject --> brain_region, brain_region --> channel number
                #does it use eib or spike?
            #parameter --> xlsx or csv TO DICTIONARY used as attribute for recording

            return self.events
        

class LFPrecordingCollection:
    # test case = one recording (rewarded, omission)
    # event input = dict: values = start, stop
    #map event type to all start and stop times
        #unique() of event types to get events

    """
    reward non reward
    social and nonsocial

    4 event types: [win, loss, reward, omission]
    win (on excel) --> if subject is also a condition --> win
        subject info = subject --> if condition == subject --> win

    NOTE: ignore competition closeness

    reward social --> win
    reward non social --> reward
    non reward social --> loss
    non reward non social --> omission

    read from dropbox api

    compare attributes of megans class with this class
        and then check if attributes are necessary for LFP

    """


    #tonetimes path is behavior dictionary that the user has to make
    #index is brain region from channel map
    #all brain regions use multi-dimensional numpy array
    #frequency is a filter on the channel map brain region

    def __init__(self, path, channel_map_path, sampling_rate=1000):
        self.path = path #path to data folder (recording folder from trodes) --> data/omissision/test
        self.sampling_rate = sampling_rate

        self.channel_map_df = pd.read_excel(channel_map_path)
        self.make_collection()


    def make_collection(self):
        collection = {}
        #read excel file
        data = pd.read_excel(self.path)
        #loop through test_1_merged.rec given reware_competition_extention dir
        for root, dirs, files in os.walk(self.path):
            for directory in dirs:
                if directory.endswith("merged.rec"):
                    #getting files in that directory
                    for file in os.listdir(os.path.join(root, directory)):
                        #call recording object
                        recording = LFPRecording(os.path.join(root, directory, file))
                        #add to collection
                        collection[directory] = recording
        self.collection = collection

testData = LFPrecordingCollection("test.xlsx")

'''
TEST THIS CODE:

current_recording = se.read_spikegadgets(recording_path, stream_id=ECU_STREAM_ID)
current_recording = se.read_spikegadgets(recording_path, stream_id=TRODES_STREAM_ID)
print(recording_basename)
# Preprocessing the LFP
current_recording = sp.bandpass_filter(current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
current_recording = sp.notch_filter(current_recording, freq=ELECTRIC_NOISE_FREQ)
current_recording = sp.resample(current_recording, resample_rate=LFP_SAMPLING_RATE)
current_recording = sp.zscore(current_recording)

'''