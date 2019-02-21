"""
Optical tweezer calibration functions
"""
from __future__ import absolute_import, print_function, division
from tweezer.conf import TweezerConfig
from tweezer.progress_bar import print_progress_bar
import time
import numpy as np

def function1(a, b = 0.):
    """A short description.
    
    A detailed description of this function. What it does and where it is used.
    Numpy documentation style.
    
    Parameters
    ----------
    a : float
        First parameter
    b : int or float, optional
        Second parameter
    
    Returns
    -------
    array : ndarray
        Computed array.
        
    Examples
    --------
    If it is possible to have a doctest example of this function please write
    them
    
    >>> function1((1.,2.),1.) # doctest: +NORMALIZE_WHITESPACE
    array([2., 3.])
    
    """
    if TweezerConfig.verbose > 0:
        print("Computing stuff")
    for i in range(100):
        print_progress_bar(i,100, level = TweezerConfig.verbose)
        time.sleep(0.04)
    print_progress_bar(100,100, level = TweezerConfig.verbose)
    return np.array(a) + b

if __name__ == "__main__":
    import doctest
    doctest.testmod()