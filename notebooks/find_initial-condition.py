import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from pyIEEM.data.data import get_hospitalisation_incidence
from pyIEEM.models.utils import initialize_model

# settings
country = 'SWE'
nrows=3
ncols=4
simulation_start = '2020-02-01'
calibration_end = '2020-03-22'

# get data
data = get_hospitalisation_incidence(country)

# setup model
age_classes = pd.IntervalIndex.from_tuples([(0,5),(5,10),(10,15),(15,20),(20,25),(25,30),(30,35),(35,40),(40,45),(45,50),(50,55),(55,60),(60,65),(65,70),(70,75),(75,80),(80,120)], closed='left')
#model = initialize_model(country, age_classes, True, simulation_start)

# adjust initial condition

# simulate
#out = model.sim([simulation_start, simulation_end])

# visualize data and model prediction
dates = data.index.get_level_values('date').unique()
spatial_units = data.index.get_level_values('spatial_unit').unique()
counter = 0
while counter <= len(spatial_units):
    fig,axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(11.7, 8.3), sharex=True)
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        if i+counter <= len(spatial_units):
            if i+ counter < len(spatial_units):
                # plot data
                ax.scatter(dates, data.loc[slice(None), spatial_units[i+counter]], edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                # set title
                ax.set_title(spatial_units[i+counter])                
            else:
                # plot data
                ax.scatter(dates, data.groupby(by='date').sum(), edgecolors='black', facecolors='white', marker='o', s=10, alpha=0.8)
                # set title
                ax.set_title(country)
            # set maximum number of labels
            ax.xaxis.set_major_locator(MaxNLocator(5))
            # rotate labels
            for tick in ax.get_xticklabels():
                tick.set_rotation(60)
        else:
            fig.delaxes(ax)

    counter += nrows*ncols
    plt.show()
    plt.close()

