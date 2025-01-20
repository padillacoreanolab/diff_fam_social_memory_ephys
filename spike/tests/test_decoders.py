import unittest
import numpy as np
import spike_analysis.decoders as decoders
import spike_analysis.spike_collection as spike


class Test_decoders(unittest.TestCase):
    def test_indices(self):
        test_list = ['a', 'a','a','a', 1, 1, 1, 1, 'b', 'b','b']
        indices = decoders.get_indices(test_list)
        self.assertEqual(indices, [(0,3),(4,7), (8,10)])

    def test_pca_matrix(self):
        
