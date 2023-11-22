import pandas as pd

#assuming condition col = win/ loss --> unique --> all event types
#assuming subject_info is the subject attributed in the individual df

class LFPRecording:
    def __init__(self, df, record):
        #suject at that record
        self.subject = df["subject_info"][record]
        self.event_dict = {}
        self.event_dict["start"] = df["time_stamp_index"][record]
        self.event_dict["end"] = df["end_time"][record]
        self.event_dict["condition"] = df["condition"][record]
        self.video = df["video_file"][record]

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

    def __init__(self, path_to_excel):
        self.path = path_to_excel

        #type_event: dictionary "times": [start, stop], "subject": X, "title_rec": Y

        def parse_all_trials(self, path):
            data = pd.read_excel(path)
            # assuming the "time" column is the start time
            # adding const to time column to get end time
            self.data = pd.read_excel(self.path)
            #drop all the NaN values
            self.data = self.data.dropna()
            #create new dataframe
            newDF = pd.DataFrame()
            newDF = self.data[["video_file", "condition", "time_stamp_index", "subject_info"]].copy()

            newDF["end_time"] = newDF["time_stamp_index"] + 10000

            #convert to dict
            newDF = newDF.to_dict("records")
            #print keys
            print(newDF[0].keys())
            #add the recording object to the list of all recordings
        parse_all_trials(self, self.path)

testData = LFPrecordingCollection("test.xlsx")
