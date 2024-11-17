import numpy as np
import unittest
import os
import lfp_analysis.preprocessor as preprocessor
import lfp_analysis.power as power

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745


class test_lfp_recording_power(unittest.TestCase):
    def test_calculate_power(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        rms_traces, brain_regions = preprocessor.preprocess(all_traces_arr, SUBJECT_DICT, 0.2, 0.195)
        print(rms_traces.shape)
        power_signal, frequencies = power.calculate_power(rms_traces, 200, 2, 1, 0.5)
        print(frequencies)
        # power = [samples, frequenceis, signals]
        # signals = traces.shape[0] or how many brain regions
        # frequencies = sampling_frequency * time_window_duration * time_window_step
        # samples =
        self.assertEqual(power_signal.shape, (25, 50, 5))
