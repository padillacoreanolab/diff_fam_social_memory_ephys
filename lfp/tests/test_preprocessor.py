import numpy as np
import unittest
from bidict import bidict
import lfp.lfp_analysis.preprocessor as preprocessor


class TestMapToRegion(unittest.TestCase):
    def setUp(self):
        # vHPC=1 (lowest), BLA=3, mPFC=5 (highest)
        self.subject_dict = {"mPFC": 5, "vHPC": 1, "BLA": 3}

    def test_returns_bidict(self):
        brain_region_dict, _ = preprocessor.map_to_region(self.subject_dict)
        self.assertIsInstance(brain_region_dict, bidict)

    def test_returns_all_regions(self):
        brain_region_dict, _ = preprocessor.map_to_region(self.subject_dict)
        self.assertCountEqual(brain_region_dict.keys(), ["mPFC", "vHPC", "BLA"])

    def test_channels_sorted_ascending(self):
        _, sorted_channels = preprocessor.map_to_region(self.subject_dict)
        self.assertEqual(sorted_channels, sorted(sorted_channels))

    def test_indices_are_zero_based_and_sequential(self):
        brain_region_dict, _ = preprocessor.map_to_region(self.subject_dict)
        indices = sorted(brain_region_dict.values())
        self.assertEqual(indices, list(range(len(self.subject_dict))))

    def test_lowest_channel_gets_index_zero(self):
        # vHPC has channel 1, the lowest
        brain_region_dict, _ = preprocessor.map_to_region(self.subject_dict)
        self.assertEqual(brain_region_dict["vHPC"], 0)

    def test_highest_channel_gets_last_index(self):
        # mPFC has channel 5, the highest
        brain_region_dict, _ = preprocessor.map_to_region(self.subject_dict)
        self.assertEqual(brain_region_dict["mPFC"], 2)

    def test_bidict_is_invertible(self):
        brain_region_dict, _ = preprocessor.map_to_region(self.subject_dict)
        for region, idx in brain_region_dict.items():
            self.assertEqual(brain_region_dict.inverse[idx], region)

    def test_single_region(self):
        brain_region_dict, sorted_channels = preprocessor.map_to_region({"mPFC": 7})
        self.assertEqual(brain_region_dict["mPFC"], 0)
        self.assertEqual(sorted_channels, [7])

    def test_shared_channel_gives_consistent_index(self):
        # Adding more regions should not change BLA's index relative to its channel rank
        dict_small = {"BLA": 3, "vHPC": 1}
        dict_large = {"mPFC": 5, "BLA": 3, "vHPC": 1}
        brd_small, _ = preprocessor.map_to_region(dict_small)
        brd_large, _ = preprocessor.map_to_region(dict_large)
        # BLA is channel 3, vHPC is channel 1 — BLA should be index 1 in both
        self.assertEqual(brd_small["BLA"], 1)
        self.assertEqual(brd_large["BLA"], 1)



class TestZscore(unittest.TestCase):
    def test_shape_preserved(self):
        traces = np.random.randn(100, 5)
        result = preprocessor.zscore(traces)
        self.assertEqual(result.shape, traces.shape)

    def test_median_of_output_is_zero(self):
        # subtracting the median means the output median should be 0 per column
        np.random.seed(0)
        traces = np.random.randn(500, 3) + 10  # shift away from 0 to make it non-trivial
        result = preprocessor.zscore(traces)
        medians = np.median(result, axis=0)
        np.testing.assert_array_almost_equal(medians, np.zeros(3), decimal=10)

    def test_known_values(self):
        # column [1, 2, 3, 4, 5]: median=3, MAD=1
        # zscore[0] = 0.6745 * (1-3) / 1 = -1.349
        # zscore[2] = 0.6745 * (3-3) / 1 = 0.0
        # zscore[4] = 0.6745 * (5-3) / 1 = +1.349
        traces = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = preprocessor.zscore(traces)
        self.assertAlmostEqual(result[0, 0], 0.6745 * (1 - 3) / 1)
        self.assertAlmostEqual(result[2, 0], 0.0)
        self.assertAlmostEqual(result[4, 0], 0.6745 * (5 - 3) / 1)

    def test_multiplier_is_applied(self):
        # Without the 0.6745 multiplier, result[0] would be -2.0 for column [1,2,3,4,5]
        # With it, result[0] should be -1.349
        traces = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
        result = preprocessor.zscore(traces)
        self.assertNotAlmostEqual(result[0, 0], -2.0)
        self.assertAlmostEqual(result[0, 0], -1.349, places=3)

    def test_multiple_columns_independent(self):
        # Each column should be zscored independently
        col1 = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        col2 = col1 * 100  # same shape, different scale — should give same zscores
        traces = np.column_stack([col1, col2])
        result = preprocessor.zscore(traces)
        np.testing.assert_array_almost_equal(result[:, 0], result[:, 1])


class TestZscoreFilter(unittest.TestCase):
    def setUp(self):
        self.zscores = np.array([[-5.0, -1.0, 2.0, 3.0, 4.0, 5.0],
                                  [-5.0, -1.0, 2.0, 3.0, 4.0, 5.0]])
        self.voltage = np.array([[-0.975, -0.195, 0.39, 0.585, 0.78, 0.975],
                                  [-0.975, -0.195, 0.39, 0.585, 0.78, 0.975]])
        self.threshold = 3

    def test_shape_preserved(self):
        result = preprocessor.zscore_filter(self.zscores, self.voltage, self.threshold)
        self.assertEqual(result.shape, self.zscores.shape)

    def test_values_above_threshold_are_nan(self):
        result = preprocessor.zscore_filter(self.zscores, self.voltage, self.threshold)
        # zscore=5.0, |5.0| >= 3 → NaN
        self.assertTrue(np.isnan(result[0, 5]))
        self.assertTrue(np.isnan(result[1, 5]))

    def test_values_below_negative_threshold_are_nan(self):
        result = preprocessor.zscore_filter(self.zscores, self.voltage, self.threshold)
        # zscore=-5.0, |-5.0| >= 3 → NaN
        self.assertTrue(np.isnan(result[0, 0]))
        self.assertTrue(np.isnan(result[1, 0]))

    def test_value_exactly_at_threshold_is_nan(self):
        # mask is |zscore| < threshold, so exactly at threshold → NaN
        result = preprocessor.zscore_filter(
            np.array([[3.0]]), np.array([[0.5]]), threshold=3
        )
        self.assertTrue(np.isnan(result[0, 0]))

    def test_values_below_threshold_preserved_from_voltage(self):
        result = preprocessor.zscore_filter(self.zscores, self.voltage, self.threshold)
        # zscore=-1.0, |-1.0| < 3 → preserves voltage value -0.195
        self.assertAlmostEqual(result[0, 1], -0.195)
        # zscore=2.0, |2.0| < 3 → preserves voltage value 0.39
        self.assertAlmostEqual(result[0, 2], 0.39)

    def test_output_comes_from_voltage_not_zscore(self):
        # Non-NaN output values should equal voltage_scaled, not zscore values
        zscores = np.array([[0.5]])
        voltage = np.array([[99.0]])
        result = preprocessor.zscore_filter(zscores, voltage, threshold=3)
        self.assertAlmostEqual(result[0, 0], 99.0)

    def test_threshold_zero_makes_everything_nan(self):
        # |zscore| is never strictly < 0
        result = preprocessor.zscore_filter(
            np.array([[0.1, 0.2]]), np.array([[1.0, 2.0]]), threshold=0
        )
        self.assertTrue(np.all(np.isnan(result)))

    def test_high_threshold_preserves_everything(self):
        result = preprocessor.zscore_filter(self.zscores, self.voltage, threshold=100)
        np.testing.assert_array_equal(result, self.voltage)


class TestScaleVoltage(unittest.TestCase):
    def test_shape_preserved(self):
        traces = np.random.randn(100, 5)
        result = preprocessor.scale_voltage(traces, 0.195)
        self.assertEqual(result.shape, traces.shape)

    def test_multiplies_by_scalar(self):
        traces = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = preprocessor.scale_voltage(traces, 0.195)
        np.testing.assert_array_almost_equal(result, traces * 0.195)

    def test_zero_scaling_gives_zeros(self):
        traces = np.random.randn(10, 3)
        result = preprocessor.scale_voltage(traces, 0.0)
        np.testing.assert_array_equal(result, np.zeros_like(traces))

    def test_unit_scaling_is_identity(self):
        traces = np.random.randn(50, 4)
        result = preprocessor.scale_voltage(traces, 1.0)
        np.testing.assert_array_equal(result, traces)

    def test_negative_values_stay_negative(self):
        traces = np.array([[-1.0, -2.0]])
        result = preprocessor.scale_voltage(traces, 0.195)
        self.assertTrue(np.all(result < 0))


class TestRootMeanSquare(unittest.TestCase):
    def test_shape_preserved(self):
        traces = np.random.randn(100, 5)
        result = preprocessor.root_mean_square(traces)
        self.assertEqual(result.shape, traces.shape)

    def test_output_has_unit_rms_per_column(self):
        # nanmean(output**2) per column should equal 1 after normalization
        np.random.seed(7)
        traces = np.random.randn(500, 4) * 10
        result = preprocessor.root_mean_square(traces)
        rms_per_col = np.nanmean(result ** 2, axis=0)
        np.testing.assert_array_almost_equal(rms_per_col, np.ones(4))

    def test_nan_positions_are_preserved(self):
        traces = np.array([[1.0, np.nan],
                           [2.0, np.nan],
                           [3.0, 4.0]])
        result = preprocessor.root_mean_square(traces)
        self.assertTrue(np.isnan(result[0, 1]))
        self.assertTrue(np.isnan(result[1, 1]))
        self.assertFalse(np.isnan(result[2, 1]))

    def test_all_nan_column_stays_nan(self):
        traces = np.array([[np.nan, 1.0],
                           [np.nan, 2.0],
                           [np.nan, 3.0]])
        result = preprocessor.root_mean_square(traces)
        self.assertTrue(np.all(np.isnan(result[:, 0])))
        self.assertFalse(np.any(np.isnan(result[:, 1])))

    def test_unit_rms_ignores_nans_in_normalization(self):
        # NaN values should be excluded from the RMS computation but stay NaN in output
        traces = np.array([[1.0], [2.0], [np.nan], [3.0], [4.0]])
        result = preprocessor.root_mean_square(traces)
        self.assertTrue(np.isnan(result[2, 0]))
        non_nan_rms = np.nanmean(result ** 2)
        self.assertAlmostEqual(non_nan_rms, 1.0)


class TestPreprocess(unittest.TestCase):
    def test_shape_preserved(self):
        np.random.seed(42)
        traces = np.random.randn(500, 5)
        result = preprocessor.preprocess(traces, threshold=4, scaling=0.195)
        self.assertEqual(result.shape, traces.shape)

    def test_output_has_unit_rms_for_valid_columns(self):
        np.random.seed(0)
        traces = np.random.randn(1000, 3)
        result = preprocessor.preprocess(traces, threshold=4, scaling=0.195)
        for col in range(result.shape[1]):
            if not np.all(np.isnan(result[:, col])):
                rms = np.nanmean(result[:, col] ** 2)
                self.assertAlmostEqual(rms, 1.0, places=5)

    def test_extreme_outlier_becomes_nan(self):
        np.random.seed(1)
        traces = np.random.randn(500, 2)
        traces[250, 0] = 1e6  # obvious spike
        result = preprocessor.preprocess(traces, threshold=4, scaling=0.195)
        self.assertTrue(np.isnan(result[250, 0]))

    def test_all_nan_column_stays_nan(self):
        np.random.seed(2)
        traces = np.random.randn(200, 3)
        traces[:, 1] = np.nan
        result = preprocessor.preprocess(traces, threshold=4, scaling=0.195)
        self.assertTrue(np.all(np.isnan(result[:, 1])))

    def test_scaling_does_not_change_nan_mask(self):
        # zscore is scale-invariant, so different scalings should produce the same NaN positions
        np.random.seed(5)
        traces = np.random.randn(300, 2)
        result1 = preprocessor.preprocess(traces, threshold=4, scaling=1.0)
        result2 = preprocessor.preprocess(traces, threshold=4, scaling=2.0)
        np.testing.assert_array_equal(np.isnan(result1), np.isnan(result2))


if __name__ == "__main__":
    unittest.main()
