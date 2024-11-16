import os
import numpy as np
import unittest

SUBJECT_DICT = {1.1: {'mPFC': 19.0, 'vHPC': 31.0, 'BLA': 30.0, 'NAc': 28.0, 'MD': 29.0},
                1.2: {'mPFC': 26.0, 'vHPC': 31.0, 'BLA': 30.0, 'NAc': 28.0, 'MD': 29.0},
                1.3: {'mPFC': 9.0, 'vHPC': 31.0, 'BLA': 30.0, 'NAc': 28.0, 'MD': 29.0}}

class test_lfp_recording_preprocessing(unittest.TestCase):

    def test_map_to_region(self):
        traces = os.path.join('test_data', 'test_traces.csv')
        traces_arr = np.array(traces)
        self.assertIs(type(traces_arr), np.ndarray)

        

