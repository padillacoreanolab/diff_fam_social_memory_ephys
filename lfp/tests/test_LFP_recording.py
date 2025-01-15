import unittest
import os
from lfp_analysis.LFP_recording import LFPRecording
import numpy.testing as npt
from pathlib import Path
from tests.utils import EXAMPLE_RECORDING_FILEPATH

CHANNEL_DICT = {"mPFC": 1, "vHPC": 9, "BLA": 11, "NAc": 27, "MD": 3}


def helper():
    lfp_rec = LFPRecording("test subject", {}, CHANNEL_DICT, EXAMPLE_RECORDING_FILEPATH, "")
    return lfp_rec


def helper_cups():
    filepath = "/Volumes/SheHulk/cups/data/11_cups_p4.rec/11_cups_p4_merged.rec"
    lfp_rec = LFPRecording("test subject", {}, CHANNEL_DICT, filepath)
    return lfp_rec


class TestLFPRecording(unittest.TestCase):
    def test_read_trodes(self):
        lfp_rec = helper()
        self.assertIsNotNone(lfp_rec.recording)
        traces = lfp_rec._get_selected_traces()
        self.assertEqual(traces.shape[0], len(CHANNEL_DICT))

    def test_channel_order(self):
        lfp_rec_0 = LFPRecording(
            "test subject 1", {}, {"mPFC": 1, "BLA": 7, "vHPC": 31}, EXAMPLE_RECORDING_FILEPATH, ""
        )
        traces_0 = lfp_rec_0._get_selected_traces()

        lfp_rec_1 = LFPRecording("test subject 2", {}, {"BLA": 7}, EXAMPLE_RECORDING_FILEPATH, "")
        traces_1 = lfp_rec_1._get_selected_traces()
        npt.assert_array_equal(traces_0[1], traces_1[0])


class TestLFPRecordingTimestampsFile(unittest.TestCase):
    def test_can_find_timestamps_file(self):
        lfp_rec = helper()
        lfp_rec.find_start_recording_time()

        self.assertIsNotNone(lfp_rec.first_timestamp)
        self.assertEqual(lfp_rec.first_timestamp, 800146)

    def test_get_traces_sets_timestamp(self):
        lfp_rec = helper()
        traces_before_trim = lfp_rec.recording.get_traces().T

        lfp_rec._get_selected_traces()

        self.assertIsNotNone(lfp_rec.first_timestamp)
        self.assertEqual(lfp_rec.first_timestamp, 800146)
        lfp_rec.traces
        self.assertIsNotNone(lfp_rec.traces)
        self.assertTrue(lfp_rec.traces.shape[0] < traces_before_trim.shape[0])
