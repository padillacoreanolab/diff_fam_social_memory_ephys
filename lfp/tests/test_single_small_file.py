import os
import numpy as np
import unittest
import lfp_analysis.mapping as mapping

SUBJECT_DICT = {'mPFC': 19.0, 'vHPC': 31.0, 'BLA': 30.0, 'NAc': 28.0, 'MD': 29.0}
            

class test_lfp_recording_preprocessing(unittest.TestCase):

    def test_map_to_region(self):
        traces = os.path.join('tests', 'test_data', 'test_traces.csv')
        traces_arr = np.loadtxt(traces, delimiter=',')
        self.assertIs(type(traces_arr), np.ndarray)
        actual_output = mapping.map_to_region(traces_arr, SUBJECT_DICT)
        self.assertEqual(actual_output.keys(), SUBJECT_DICT.keys())
        
    
