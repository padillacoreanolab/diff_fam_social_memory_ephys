import numpy as np
import unittest
import os
import lfp_analysis.preprocessor as preprocessor
import lfp_analysis.connectivity_wrapper as connectivity_wrapper
from tests import utils

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745


class test_lfp_recording_power(unittest.TestCase):
    def test_calculate_all_connectivity_small_file(self):
        traces = utils.load_test_traces()
        rms_traces = preprocessor.preprocess(traces, 0.2, 0.195)

        connectivity, frequencies, power, coherence, granger = connectivity_wrapper.connectivity_wrapper(
            rms_traces, 200, 2, 1, 0.5
        )

        self.assertEqual(power.shape, (24, 100, 5))
        self.assertEqual(coherence.shape, (24, 100, 5, 5))
