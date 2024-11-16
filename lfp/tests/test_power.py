import numpy as np
import matplotlib.pyplot as plt
from bidict import bidict
import scipy.stats as stats
from spectral_connectivity import Multitaper, Connectivity
import lfp_analysis.preprocessor as preprocessor
import lfp_analysis.power as power
import shutil
import unittest

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745

class test_lfp_recording_power(unittest.TestCase):
    def calculate_power(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        power, frequencies = preprocessor.preprocess(all_traces_arr, SUBJECT_DICT)
        self.as
