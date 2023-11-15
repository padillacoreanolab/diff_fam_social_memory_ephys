import pandas as pd
class LFPrecordingCollection:
    def __init__(self, path_to_excel):
        self.path = path_to_excel

        def parse_all_trials(self, path):
            data = pd.read_excel(path)
            # assuming the "time" column is the start time
            # adding const to time column to get end time
            self.data = pd.read_excel(self.path)
            #iterate through unique video_files and make a new dataframe for each
            tempDF = pd.DataFrame()
            #add unique video files as column in dataframe
            tempDF["video_file"] = self.data["video_file"].unique()
            #iterate through original data to add event, start, end to new dataframe
            for index, row in self.data.iterrows():
                #add event
                tempDF["event"] = row["event"]
                #add start time
                tempDF["start_time"] = row["time_stamp"]
                #add end time
                tempDF["end_time"] = row["end_time"]

            # convert time to int
            self.data["time_stamp"] = self.data["time_stamp"].astype(int)
            # adding end time column
            # TODO: ask albert if end time exists
            self.data["end_time"] = self.data["time_stamp"] + 10000

test = pd.read_excel("rce_tone_timestamp.xlsx")
print(test["time_stamp_index"])