import pandas as pd

#assuming condition col = win/ loss --> unique --> all event types
#assuming subject_info is the subject attributed in the individual df

class LFPrecordingCollection:
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
        parse_all_trials(self, self.path)

testData = LFPrecordingCollection("rce_tone_timestamp.xlsx")
