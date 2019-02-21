"""
Optical tweezer calibration functions
"""

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
    return np.array(a) + b

if __name__ == "__main__":
    import doctest
    doctest.testmod()