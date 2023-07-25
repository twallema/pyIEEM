import pandas as pd
import numpy as np

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

def convert_age_stratified_quantity(data, age_classes, demography):
        """ 
        Given a pd.Series containing age classes and some quantity: "[age_group_lower, age_group_upper] : quantity",
        this function can convert the data into different age groups using demographic weighing

        input
        =====

        data: pd.Series
            a dataset containing a quantity in age bands. Index must be of type pd.Intervalindex.
        
        age_classes : pd.IntervalIndex
            desired age groups of output pd.Series.

        demography: pd.Series
            demography of the country under study. Index must contain the number of individuals per year of age (type: float). 

        output
        ======

        out: pd.Series
            A dataset containing the quantity in the desired age bands.
        """
        
        # check if data and demography are pd.Series
        assert isinstance(data, pd.Series)
        assert isinstance(demography, pd.Series)
        # Pre-allocate new series
        out = pd.Series(index=age_classes, dtype=float)
        # Format demographics into desired age classes
        demo_format = demography.groupby(pd.cut(demography.index.values, data.index)).sum()#.squeeze() --> if only one age group this becomes an int --> always excepts
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

def make_reciprocal(matrix, demography):
    """
    A function to make a contact matrix reciprocal using demographic weighing

    input
    =====

    matrix: pd.Series
        contact matrix given as pd.Series with a multiindex containing two levels ('age_x' and 'age_y').
        the first level is assumed to be the x-axis (survey participant) and the second level the y-axis (contacted individual)

    demography: pd.Series
        demography of the country under study. Index must contain the number of individuals per year of age (type: float). 

    output
    ======

    out: pd.Series
        contact matrix in desired age classes
    """

    # Verify matrix is square
    assert all(matrix.index.get_level_values(0).unique().values == matrix.index.get_level_values(1).unique().values)

    # Convert demography to age classes
    age_classes = matrix.index.get_level_values(0).unique()
    desired_demography = demography.groupby(pd.cut(demography.index.values, age_classes)).sum()

    # Loop over every row
    c = np.zeros([len(age_classes), len(age_classes)])
    for x, age_class_x in enumerate(age_classes):
        for y, age_class_y in enumerate(age_classes):
            N_x = desired_demography.loc[age_class_x]
            N_y = desired_demography.loc[age_class_y]
            m_xy = matrix.loc[age_class_x, age_class_y]
            m_yx = matrix.loc[age_class_y, age_class_x]
            c[x, y] = (m_xy*N_x + m_yx*N_y)/(2*N_x*N_y)*N_x

    # Assign data to matrix
    matrix = matrix.to_frame()
    matrix.loc[(slice(None), slice(None)),matrix.columns] = c.flatten()
    matrix = matrix.squeeze()
    return matrix

def aggregate_contact_matrix(matrix, age_classes, demography):
    """
    A function to convert a (square) contact matrix from a given to a desired set of age classes using demographic weighing

    input
    =====

    matrix: pd.Series
        contact matrix given as pd.Series with a multiindex containing two levels ('age_x' and 'age_y').
        the first level is assumed to be the x-axis (survey participant) and the second level the y-axis (contacted individual)

    age_classes : pd.IntervalIndex
        desired age groups of contact matrix

    demography: pd.Series
        demography of the country under study. Index must contain the number of individuals per year of age (type: float). 

    output
    ======

    out: pd.Series
        contact matrix in desired age classes

    remarks
    =======

    Only works if the minimum and maximum ages of the demography, x-axis and y-axis of the contact matrix are identical!
    """
    ## assert matrix and demography are pd.Series
    assert isinstance(matrix, pd.Series)
    assert isinstance(demography, pd.Series)
    ## assert minimum and maximum ages are identical
    assert matrix.index.get_level_values(0).unique().min().left == matrix.index.get_level_values(1).unique().min().left
    assert matrix.index.get_level_values(0).unique().max().right == matrix.index.get_level_values(1).unique().max().right
    assert demography.index.max() == matrix.index.get_level_values(0).unique().max().right
    assert demography.index.min() == matrix.index.get_level_values(0).unique().min().left
    assert matrix.index.get_level_values(0).unique().max().right == age_classes.max().right
    ## given age classes
    given_age_classes = matrix.index.get_level_values(0).unique().values
    desired_age_classes = age_classes
    ## pre-allocate output dataframe
    out = pd.Series(index=pd.MultiIndex.from_product([desired_age_classes, desired_age_classes], names=matrix.index.names), dtype=float)
    out.name = matrix.name
    ## convert demography in desired_age_classes
    desired_demography = demography.groupby(pd.cut(demography.index.values, desired_age_classes)).sum().squeeze()
    ## loop over age_x, convert age_y to desired age classes
    converted_age_y = []
    for age_x in given_age_classes:
        converted_age_y.append(convert_age_stratified_quantity(matrix.loc[age_x], desired_age_classes, demography).values) 
    ## loop over desired age_x
    for age_class in desired_age_classes:
        result = np.zeros(len(age_classes), dtype=float)
        for i in range(age_class.left, age_class.right):
            # fraction of population in desired age class currently of age i
            f = (demography.loc[i]/desired_demography.loc[desired_age_classes.contains(i)]).values[0]
            # number of contacts of age i (in given age class)
            n = converted_age_y[np.where(given_age_classes.contains(i))[0][0]]
            # multiply and sum
            result += f*n
        # save result
        out.loc[age_class, slice(None)] = result
    return out

import statsmodels.api as sm
from statsmodels.gam.api import GLMGam, BSplines
def smooth_contact_matrix(matrix, df, degree):
    """
    A function to GAM smooth contact matrices
    
    Input
    =====
    
    matrix: pd.Series
        contact matrix given as pd.Series with a multiindex containing two levels ('age_x' and 'age_y')
        the first level is assumed to be the x-axis (survey participant) and the second level the y-axis (contacted individual)

    df: int
        Number of B-splines for GAM fit
        
    degree: int
        Degree of B-splines for GAM fit    
    
    Output
    ======
    
    matrix: pd.Series
        Smoothed contact matrix. Index: age_x, age_y. Name: contacts. 
    """
    
    # name of endogeneous variable
    y = f'{matrix.name} ~'
    # extract age classes and midpoints
    age_classes_x = matrix.index.get_level_values(0).unique().values
    age_classes_y = matrix.index.get_level_values(1).unique().values
    midpoints_x = [age_class.mid for age_class in age_classes_x]
    midpoints_y = [age_class.mid for age_class in age_classes_y]
    index_names = matrix.index.names
    # replace index with midpoints
    matrix.index = matrix.index.set_levels(midpoints_x, level=0)
    matrix.index = matrix.index.set_levels(midpoints_y, level=1)
    matrix.index.names = ['x', 'y']
    # drop index
    matrix = matrix.reset_index()
    matrix = matrix.astype(float)
    # construct GAM model
    x_spline = matrix[['x', 'y']]
    bs = BSplines(x_spline, df=[df,df], degree=[degree, degree])
    model = GLMGam.from_formula(y +'x + y + x:y', data=matrix, smoother=bs,
                                family=sm.families.NegativeBinomial(), alpha=np.array([0.1, 0.1]), method='newton')
    # fit GAM model
    res = model.fit()
    # predict matrix of contacts
    smoothed_values = res.predict()
    # merge back into dataframe
    matrix['smoothed_contacts'] = smoothed_values
    # set multiindex
    matrix = matrix.groupby(by=['x','y']).last()
    # re-introduce the age classes as index
    matrix.index = matrix.index.set_levels(age_classes_x, level=0)
    matrix.index = matrix.index.set_levels(age_classes_y, level=1)
    # set names back to original
    matrix.index.names = index_names
    return matrix.squeeze()