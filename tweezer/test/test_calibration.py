"""Unit tests for the calibration module"""

import unittest

import numpy as np
#import calibration
import tweezer.calibration as calibration
import generate_calibration_data

class TestCalibration(unittest.TestCase):
    
    def setUp(self):
        self.kx = 1e-6
        self.ky = 2e-6
        self.phi = 0 
        self.expected_result = np.array([tuple([self.kx, self.ky]), self.phi])
        
    def test_center_and_rotate(self):
        data =  generate_calibration_data.generate(
                [self.kx, self.ky], phi = self.phi
                )
        result = calibration.center_and_rotate(
                        data[1], data[2]
                        )
        self.assertTrue(np.allclose(result[2], self.expected_result[1], 
                                    atol = 0.1))
        
    def test_running_average(self):
        data =  generate_calibration_data.generate(
                [self.kx, self.ky], phi = self.phi
                )
        x = calibration.subtract_running_average(data[0], data[1])
        y = calibration.subtract_running_average(data[0], data[2])
        self.assertTrue(
                np.allclose(np.mean(x), 0, atol = 1e-3) and
                np.allclose(np.mean(y), 0, atol = 1e-3)
                )
        #We could also check the standard deviation.
        
    def test_calibrate(self):
        data =  generate_calibration_data.generate(
                [self.kx, self.ky], phi = self.phi
                )
        result = calibration.calibrate(
                        data[0], data[1], data[2]
                        )
        self.assertTrue(
                np.allclose(result[0], self.expected_result[0], atol=1e-6) and
                np.allclose(result[1], self.expected_result[1], atol=0.1))
        

if __name__ == "__main__":
    unittest.main()
