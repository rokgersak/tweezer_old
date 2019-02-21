"""Unit tests for the calibration module"""

import unittest

import numpy as np
import tweezer.calibration as calib

class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.parameter = 1.
        self.expected_result = np.array([2.,3.])
        
    def test_function1(self):
        result = calib.function1((1,2),b = 1)
        self.assertTrue(np.allclose(result, self.expected_result)) 

if __name__ == "__main__":
    unittest.main()