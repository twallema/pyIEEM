import pandas as pd

def to_pd_interval(istr, dtype=int):
    """
    A function to convert a string representation of a pd.Interval to pd.Interval. Bounding values are converted to integers.

    Input
    -----

    istr: str
        String representation of a pd.IntervalIndex, f.i. '[0,20('

    dtype: obj
        Desired datatype of interval bounds. Defaults to integers.

    Example use
    -----------
    
    # Convert the string representation of an interval in the column 'age' to a pd.IntervalIndex
    pd.read_csv('my_data.csv', converters={'age': to_pd_interval})
    """

    c_left = istr[0]=='['
    c_right = istr[-1]==']'
    closed = {(True, False): 'left',
                (False, True): 'right',
                (True, True): 'both',
                (False, False): 'neither'}[c_left, c_right]
    # Converts bounds to integers! Change to float if necessary!
    left, right = map(dtype, istr[1:-1].split(','))
    return pd.Interval(left, right, closed)