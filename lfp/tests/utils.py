import numpy as np
import os
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp


def load_test_traces():
    traces_path = os.path.join("tests", "test_data", "test_traces.csv")
    all_traces_arr = np.loadtxt(traces_path, delimiter=",")
    return all_traces_arr


def load_large_test_traces():
    traces_path = os.path.join("tests", "test_data", "11_cups_p4_merged.rec_500_100000.csv")
    all_traces_arr = np.loadtxt(traces_path, delimiter=",")
    return all_traces_arr


def create_test_data(recording_path):
    TRODES_STREAM_ID = "trodes"
    RECORDING_EXTENTION = "*merged.rec"

    LFP_FREQ_MIN = 0.5
    LFP_FREQ_MAX = 300
    ELECTRIC_NOISE_FREQ = 60
    LFP_SAMPLING_RATE = 1000
    EPHYS_SAMPLING_RATE = 20000
    start_frame = 500
    stop_frame = 100000

    print("Saving data for ", recording_path)
    current_recording = se.read_spikegadgets(recording_path, stream_id=TRODES_STREAM_ID)
    # # Preprocessing the LFP
    current_recording = sp.notch_filter(current_recording, freq=ELECTRIC_NOISE_FREQ)
    current_recording = sp.bandpass_filter(current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
    current_recording = sp.resample(current_recording, resample_rate=LFP_SAMPLING_RATE)

    filename = f"{os.path.basename(recording_path)}_{start_frame}_{stop_frame}.csv"

    traces = current_recording.get_traces(start_frame=start_frame, end_frame=stop_frame).T
    save_path = os.path.join("tests", "test_data", filename)
    np.savetxt(save_path, traces, delimiter=",")


if __name__ == "__main__":
    # external_path = r"D:\cups\data\11_cups_p4.rec\11_cups_p4_merged.rec"
    external_path = "/Volumes/SheHulk/cups/data/11_cups_p4.rec/11_cups_p4_merged.rec"
    create_test_data(external_path)
