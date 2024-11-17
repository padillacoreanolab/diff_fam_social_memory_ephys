import numpy as np
import os
import spikeinterface.extractors as se
import spikeinterface.preprocessing as sp
import lfp_analysis.preprocessor as preprocessor
import argparse


def load_test_traces():
    traces_path = os.path.join("tests", "test_data", "11_cups_p4_merged.rec_500_3000.csv")
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
    stop_frame = 3000

    print("Saving data for ", recording_path)
    current_recording = se.read_spikegadgets(recording_path, stream_id=TRODES_STREAM_ID)
    # Preprocessing the LFP
    current_recording = sp.notch_filter(current_recording, freq=ELECTRIC_NOISE_FREQ)
    current_recording = sp.bandpass_filter(current_recording, freq_min=LFP_FREQ_MIN, freq_max=LFP_FREQ_MAX)
    current_recording = sp.resample(current_recording, resample_rate=LFP_SAMPLING_RATE)

    filename = f"{os.path.basename(recording_path)}_{start_frame}_{stop_frame}.csv"

    SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
    brain_regions, sorted_channels = preprocessor.map_to_region(SUBJECT_DICT)

    traces = current_recording.get_traces(start_frame=start_frame, end_frame=stop_frame).T

    traces = traces[sorted_channels]

    save_path = os.path.join("tests", "test_data", filename)
    np.savetxt(save_path, traces, delimiter=",")


def main():
    parser = argparse.ArgumentParser(description="LFP Test Data Utility")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Load test traces command
    load_parser = subparsers.add_parser("load", help="Load test traces")
    load_parser.add_argument(
        "--large", action="store_true", help="Load large test traces instead of regular test traces"
    )

    # Create test data command
    create_parser = subparsers.add_parser("create", help="Create test data")
    create_parser.add_argument("recording_path", help="Path to the recording file")

    args = parser.parse_args()

    if args.command == "load":
        if args.large:
            traces = load_large_test_traces()
            print("Loaded large test traces with shape:", traces.shape)
        else:
            traces = load_test_traces()
            print("Loaded test traces with shape:", traces.shape)
    elif args.command == "create":
        create_test_data(args.recording_path)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
