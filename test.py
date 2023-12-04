#testing spike analysis code and saving in text file
import glob
import os
import numpy as np
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp


RECORDING_EXTENTION = '*.rec'
session_dir = "reward_competition_extention/data/omission/test/test_1_merged.rec"
ECU_STREAM_ID = "ECU"
TRODES_STREAM_ID = "trodes"
LFP_FREQ_MIN = 0.5
LFP_FREQ_MAX = 300
ELECTRIC_NOISE_FREQ = 60
LFP_SAMPLING_RATE = 1000

files = {"og": "test_outputs/original.txt",
         "bandpass": "test_outputs/bandpass.txt",
         "notch": "test_outputs/notch.txt",
         "resample": "test_outputs/resample.txt",
         "zscore": "test_outputs/zscore.txt"}

for recording_path in glob.glob(os.path.join(session_dir, RECORDING_EXTENTION)):
    # assumes subject name in file name
    try:
        recording_basename = os.path.splitext(os.path.basename(recording_path))[0]
        # checking to see if the recording has an ECU component
        # if it doesn't, then the next one be extracted
        current_recording = se.read_spikegadgets(recording_path, stream_id=ECU_STREAM_ID)
        current_recording = se.read_spikegadgets(recording_path,
                                                 stream_id=TRODES_STREAM_ID)  # we need to confer with leo what these lines do
        print("~~~~~~~~~~~~~~~ORIGINAL~~~~~~~~~~~~~~~")
        print(current_recording)
        with open(files["og"], "w") as f:
            f.write(str(current_recording))
        print(recording_basename)
        # Preprocessing the LFP
        # higher than 300 is action potential and lower than 0.5 is noise
        current_recording = sp.bandpass_filter(current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
        print("~~~~~~~~~~~~~~~AFTER BANDPASS~~~~~~~~~~~~~~~")
        print(current_recording)
        #dump output into text file and name it bandpass
        with open(files["bandpass"], "w") as f:
            f.write(str(current_recording))

        current_recording = sp.notch_filter(current_recording, freq=ELECTRIC_NOISE_FREQ)
        print("~~~~~~~~~~~~~~~AFTER NOTCH~~~~~~~~~~~~~~~")
        print(current_recording)
        with open(files["notch"], "w") as f:
            f.write(str(current_recording))
        current_recording = sp.resample(current_recording, resample_rate=LFP_SAMPLING_RATE)
        print("~~~~~~~~~~~~~~~AFTER RESAMPLE~~~~~~~~~~~~~~~")
        print(current_recording)
        with open(files["resample"], "w") as f:
            f.write(str(current_recording))
        current_recording = sp.zscore(current_recording)# zscore single because avg across animals is in same scale
        print("~~~~~~~~~~~~~~~AFTER ZSCORE~~~~~~~~~~~~~~~")
        print(current_recording)
        with open(files["zscore"], "w") as f:
            f.write(str(current_recording))
    except Exception as error:
        # handle the exception
        print("An exception occurred:", error)