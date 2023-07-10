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

import numpy as np
import sys
def convert_age_stratified_quantity(data, age_classes, demography):
        """ 
        Given an age-stratified pd.Series of some quantity: [age_group_lower, age_group_upper] : quantity,
        this function can convert the data into a user-defined age-stratification using demographic weighing

        Parameters
        ----------
        data: pd.Series
            A dataset containing a quantity in age bands. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            Desired age groups of output pd.Series.

        demography: pd.Series
            Demography of the country under study. Index must contain the number of individuals per year of age (type: float).

        Returns
        -------

        out: pd.Series
            Converted data.
        """

        # Pre-allocate new series
        out = pd.Series(index = age_classes, dtype=float)
        # Format demographics --> age_groups_data / no_individuals
        demo_format = pd.Series(0, index=data.index)
        for interval in data.index:
            count=0
            for age in range(interval.left, interval.right):
                count+= demography[age]
            demo_format.loc[interval] = count
        
        # Loop over desired intervals
        for idx,interval in enumerate(age_classes):
            result = []
            for age in range(interval.left, interval.right):
                try:
                    result.append((demography[age]/demo_format[data.index.contains(age)]*data.iloc[np.where(data.index.contains(age))[0][0]]).values)
                except:
                    result.append(0)
            out.iloc[idx] = sum(result)
        return out