import pandas as pd
import os

#assuming condition col = win/ loss --> unique --> all event types
#assuming subject_info is the subject attributed in the individual df

class LFPRecording:
    def __init__(self, path, sampling_rate=20000):
        self.path = path
        self.sampling_rate = sampling_rate
        self.events = {}




class LFPrecording:
    def __init__(self, record):
        self.event = record["event"]
        self.start = record["time_stamp_index"]
        self.end = record["end_time"]
        self.subject = record["subject_info"]
        self.video = record["video_file"]
        

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

    def __init__(self, path, tone_times_path, channel_map_path, sampling_rate=1000):
        self.path = path
        self.sampling_rate = sampling_rate
        self.tone_times_df = pd.read_excel(tone_times_path)
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
