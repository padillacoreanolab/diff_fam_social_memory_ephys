import unittest
import numpy as np
from spike.spike_analysis.firing_rate_calculations import get_spiketrain
from spike.spike_analysis.firing_rate_calculations import get_firing_rate

TIMESTAMP_ARRAY = [1, 2, 8, 9, 10, 11, 15, 16, 19, 20, 22, 30]
LAST_TIMESTAMP = 30


class Test_firing_rate_calculations(unittest.TestCase):
    def test_spiketrain(self):
        spiketrain = get_spiketrain(TIMESTAMP_ARRAY, LAST_TIMESTAMP, timebin=1, sampling_rate=1000)
        self.assertEqual(len(spiketrain), 30)
        self.assertTrue(
            (
                spiketrain == [1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1]
            ).all()
        )
        spiketrain = get_spiketrain(TIMESTAMP_ARRAY, LAST_TIMESTAMP, timebin=1, sampling_rate=2000)
        self.assertEqual(len(spiketrain), 15)
        spiketrain = get_spiketrain(TIMESTAMP_ARRAY, LAST_TIMESTAMP, timebin=2, sampling_rate=1000)
        self.assertEqual(len(spiketrain), 15)
        spiketrain = get_spiketrain(TIMESTAMP_ARRAY, LAST_TIMESTAMP, timebin=10, sampling_rate=1000)
        self.assertEqual(len(spiketrain), 3)
        self.assertTrue((spiketrain == [5, 5, 2]).all())

    def test_firing_rate(self):
        spiketrain = get_spiketrain(TIMESTAMP_ARRAY, LAST_TIMESTAMP, timebin=10, sampling_rate=1000)
        firingrates = get_firing_rate(spiketrain, timebin=10, smoothing_window=None)
        self.assertTrue((firingrates == [500, 500, 200]).all())
        spiketrain = get_spiketrain(TIMESTAMP_ARRAY, LAST_TIMESTAMP, timebin=1, sampling_rate=1000)
        firingrates = get_firing_rate(spiketrain, timebin=1, smoothing_window=10)
        self.assertEqual(len(firingrates), 30)
        firingrates_forward = get_firing_rate(spiketrain=spiketrain, timebin=1, smoothing_window=10, mode="forward")
        self.assertEqual(len(firingrates), len(firingrates_forward))
        self.assertTrue(np.isnan(firingrates_forward[0]))
        firingrates_backward = get_firing_rate(spiketrain=spiketrain, timebin=1, smoothing_window=10, mode="backward")
        self.assertTrue(np.isnan(firingrates_backward[-1]))
        self.assertEqual(len(firingrates), len(firingrates_backward))
