import os
import numpy as np
import unittest
import bidict
import lfp_analysis.preprocessor as preprocessor

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}


class test_lfp_recording_preprocessing(unittest.TestCase):

    def test_map_to_region(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        self.assertIs(type(all_traces_arr), np.ndarray)

        brain_regions, traces = preprocessor.map_to_region(all_traces_arr, SUBJECT_DICT)
        # brain regions is a bidict: {brain_region: brainregion_index}
        self.assertIs(type(brain_regions), bidict.Bidict)
        self.assertCountEqual(
            brain_regions.keys(), ["mPFC", "vHPC", "BLA", "NAc", "MD"]
        )
        # traces is numpy array [ brainregion_index, timebins ]
        self.assertIs(type(traces), np.ndarray)

        # All of the indexes in brain_regions exist in traces
        for each in brain_regions.items():
            self.assertEqual(traces[each[1]].shape, (1000,))
