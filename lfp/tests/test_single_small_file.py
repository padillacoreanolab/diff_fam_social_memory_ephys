import os
import numpy as np
import unittest
import bidict
import lfp_analysis.preprocessor as preprocessor

SUBJECT_DICT = {"mPFC": 19, "vHPC": 31, "BLA": 30, "NAc": 28, "MD": 29}
SPIKE_GADGETS_MULTIPLIER = 0.6745


class test_lfp_recording_preprocessing(unittest.TestCase):

    def test_map_to_region(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        self.assertIs(type(all_traces_arr), np.ndarray)

        brain_regions, traces = preprocessor.map_to_region(all_traces_arr, SUBJECT_DICT)
        # brain regions is a bidict: {brain_region: brainregion_index}
        self.assertIs(type(brain_regions), bidict.bidict)
        self.assertCountEqual(brain_regions.keys(), ["mPFC", "vHPC", "BLA", "NAc", "MD"])
        # traces is numpy array [ brainregion_index, timebins ]
        self.assertIs(type(traces), np.ndarray)

        # All of the indexes in brain_regions exist in traces
        for each in brain_regions.items():
            self.assertEqual(traces[each[1]].shape, (2500,))
        self.assertEqual(brain_regions["mPFC"], 0)
        self.assertEqual(brain_regions["vHPC"], 4)

    def test_zscore(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        brain_regions, traces = preprocessor.map_to_region(all_traces_arr, SUBJECT_DICT)
        # use scipy median_abs_deviation , put in 5, X array, get an array of 5 by 1
        mad_list = preprocessor.median_abs_dev(traces)
        self.assertEqual(mad_list.shape[0], traces.shape[0])
        zscore_traces = preprocessor.zscore(traces)
        self.assertEqual(traces.shape, zscore_traces.shape)

    def test_rms(self):
        traces_path = os.path.join("tests", "test_data", "test_traces.csv")
        all_traces_arr = np.loadtxt(traces_path, delimiter=",")
        brain_regions, traces = preprocessor.map_to_region(all_traces_arr, SUBJECT_DICT)
        rms_traces = preprocessor.root_mean_sqaure(traces)
        self.assertEqual(traces.shape, rms_traces.shape)
