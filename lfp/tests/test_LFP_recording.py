import unittest
import os
from lfp.lfp_analysis.LFP_recording import LFPRecording
import numpy.testing as npt
from lfp.tests.utils import EXAMPLE_RECORDING_FILEPATH
import lfp.lfp_analysis.LFP_recording as lfp_recording
import lfp.lfp_analysis.preprocessor as preprocessor
import h5py
from unittest.mock import patch
import io
import shutil
import json
import numpy as np

CHANNEL_DICT = {"mPFC": 1, "vHPC": 9, "BLA": 11, "NAc": 27, "MD": 3}
OUTPUT_DIR = os.path.join("lfp", "tests", "output")
REC_PATH = os.path.join("lfp", "tests", "output", "test")
H5_PATH = REC_PATH + ".h5"
JSON_PATH = REC_PATH + ".json"


def helper():
    lfp_rec = LFPRecording("test subject", CHANNEL_DICT, EXAMPLE_RECORDING_FILEPATH)
    return lfp_rec


# def helper_cups():
#     filepath = "/Volumes/SheHulk/cups/data/11_cups_p4.rec/11_cups_p4_merged.rec"
#     lfp_rec = LFPRecording(
#         "test subject",
#         {},
#         CHANNEL_DICT,
#         filepath,
#     )
#     return lfp_rec


class TestLFPRecording(unittest.TestCase):
    def test_read_trodes(self):
        with patch("sys.stdout", new=io.StringIO()):
            lfp_rec = helper()
            recording = lfp_rec._read_trodes()
            self.assertIsNotNone(recording)
            traces = lfp_rec._get_selected_traces(recording)
            self.assertEqual(traces.shape[1], len(CHANNEL_DICT))

    def test_channel_order(self):
        with patch("sys.stdout", new=io.StringIO()):
            lfp_rec_0 = LFPRecording(
                "test subject 1", {"mPFC": 1, "BLA": 7, "vHPC": 31}, EXAMPLE_RECORDING_FILEPATH, ""
            )
            recording_0 = lfp_rec_0._read_trodes()
            traces_0 = lfp_rec_0._get_selected_traces(recording_0)

            lfp_rec_1 = LFPRecording("test subject 2", {"BLA": 7}, EXAMPLE_RECORDING_FILEPATH, "")
            recording_1 = lfp_rec_1._read_trodes()
            traces_1 = lfp_rec_1._get_selected_traces(recording_1)
            npt.assert_array_equal(traces_0[:, 1], traces_1[:, 0])


class TestLFPRecordingTimestampsFile(unittest.TestCase):
    def test_can_find_timestamps_file(self):
        with patch("sys.stdout", new=io.StringIO()):
            lfp_rec = helper()
            lfp_rec.find_start_recording_time()
            self.assertIsNotNone(lfp_rec.first_timestamp)
            self.assertEqual(lfp_rec.first_timestamp, 800146)

    def test_get_traces_sets_timestamp(self):
        with patch("sys.stdout", new=io.StringIO()):
            lfp_rec = helper()
            recording = lfp_rec._read_trodes()
            # traces_before_trim = recording.get_traces()
            lfp_rec._get_selected_traces(recording)
            self.assertIsNotNone(lfp_rec.first_timestamp)
            self.assertEqual(lfp_rec.first_timestamp, 800146)
            self.assertIsNotNone(lfp_rec.traces)
            # some how first timestamp is after the whole recording length?
            # so need a different example recording
            # self.assertTrue(lfp_rec.traces.shape[0] < traces_before_trim.shape[0])


class TestH5File(unittest.TestCase):

    @classmethod
    def setUPClass(cls):
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.mkdir(OUTPUT_DIR)

    def test_1_save_rec(self):

        instance1 = helper()
        # instance1.process(threshold=4)
        instance1.rms_traces = preprocessor.preprocess(
            traces=instance1.traces, threshold=4, scaling=instance1.voltage_scaling
        )
        lfp_recording.LFPRecording.save_rec_to_h5(instance1, REC_PATH)
        self.assertTrue(os.path.exists(H5_PATH))
        self.assertTrue(os.path.exists(JSON_PATH))
        try:
            with h5py.File(H5_PATH, "r") as f:
                self.assertTrue(isinstance(f, h5py.File), "File is not a valid HDF5 file")
        except (ImportError, OSError) as e:
            self.fail(f"Could not verify H5 file format: {str(e)}")
        try:
            with open(JSON_PATH, "r") as f:
                json_data = json.load(f)
                self.assertTrue(isinstance(json_data, dict), "File is not a valid JSON file")
        except (json.JSONDecodeError, OSError) as e:
            self.fail(f"Could not verify JSON file format: {str(e)}")

    def test_2_load_rec(self):
        instance1 = lfp_recording.LFPRecording.load_rec_from_h5(H5_PATH)
        self.assertTrue(isinstance(instance1, lfp_recording.LFPRecording))
        self.assertTrue(instance1.subject, "subject")
        self.assertTrue(instance1.channel_map["mPFC"], "1")
        self.assertTrue(hasattr(instance1, "rms_traces"))
        self.assertFalse(hasattr(instance1, "power"))
        instance2 = helper()  # Your original instance
        instance2.rms_traces = preprocessor.preprocess(
            traces=instance2.traces, threshold=4, scaling=instance1.voltage_scaling
        )
        attrs1 = vars(instance1)
        attrs2 = vars(instance2)
        self.assertEqual(set(attrs1.keys()), set(attrs2.keys()), "Instances have different attributes")
        for attr in attrs1:
            if isinstance(attrs1[attr], np.ndarray):
                np.testing.assert_array_equal(attrs1[attr], attrs2[attr], f"Values differ for attribute {attr}")
            elif isinstance(attrs1[attr], dict):
                self.assertDictEqual(attrs1[attr], attrs2[attr], f"Dictionary values differ for attribute {attr}")
            else:
                self.assertEqual(attrs1[attr], attrs2[attr], f"Values differ for attribute {attr}")
