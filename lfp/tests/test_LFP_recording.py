import unittest
import os
from lfp_analysis.LFP_recording import LFPRecording
import numpy.testing as npt

CHANNEL_DICT = {"mPFC": 1, "vHPC": 9, "BLA": 11, "NAc": 27, "MD": 3}

EXAMPLE_REC_PATH = os.path.join("tests", "test_data", "Example_Recording", "example_recording.rec")


def helper():
    lfp_rec = LFPRecording("test subject", {}, CHANNEL_DICT, EXAMPLE_REC_PATH)
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
        lfp_rec_0 = LFPRecording("test subject 1", {}, {"mPFC": 1, "BLA": 7, "vHPC": 31}, EXAMPLE_REC_PATH)
        traces_0 = lfp_rec_0._get_selected_traces()

        lfp_rec_1 = LFPRecording("test subject 2", {}, {"BLA": 7}, EXAMPLE_REC_PATH)
        traces_1 = lfp_rec_1._get_selected_traces()
        npt.assert_array_equal(traces_0[1], traces_1[0])