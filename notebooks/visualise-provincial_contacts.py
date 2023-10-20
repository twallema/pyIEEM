import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

# settings
vacation = False
type_day = 'average'
type_contact = 'absolute_contacts'
countries = ['SWE', 'BE']
# size figure
figsize = (11.7, 8.3)
# north arrow
na_position = (0.95, 0.25)
na_xwidth = 0.115
na_yheight = 0.15
# scale
sb_positions = [(1200000, 7000000), (350000, 6300000)]
sb_lengths = [5*165000, 165000]
facs = ['   500 km', '   100 km']
numdivs = [5, 5]
lws = [5, 6]

######################
## helper functions ##
######################

def add_scalebar(ax, xy, length=50000., numdiv=5, fac="km", lw=5.,
                 marg=1000.):
    """
    Add a scalebar to the plot, using plot data units.

    Parameters
    -----------
    ax : mpl.axis
        current axis to scale bar
    xy : tuple of float
        x and y-coordinates of the lower left point of the scalebar, in map units
    lenght : float
        length in map units (m) of the scalebar
    numdiv : int
        number if division to split scalebar
    fac : str
        currently, only km for kilometer is supported
    lw : float
        width/length ratio
    marg : float
        distance between text and rectangles in map units
    """
    from itertools import cycle
    from matplotlib.patches import Rectangle

    # calculate left lower coordinates of rectangles
    x_pos = [xy[0] + i*length/numdiv for i in range(numdiv)]
    y = xy[1]
    # calculate labels
    dlabels = [int(i*length/numdiv/1000.) for i in range(numdiv + 1)]
    # put rectangles on the map
    for x, dlab, c in zip(x_pos, dlabels, cycle(['black', 'white'])):
        rect = Rectangle((x, y), length/numdiv,
                         length/numdiv/lw, facecolor=c, edgecolor='k',
                         zorder=1, clip_on=False)
        ax.add_patch(rect)

    ax.text(x_pos[-1] + length/numdiv + marg, y + length/numdiv/lw/2.,
            fac, horizontalalignment='left',
            verticalalignment='center', zorder=1, size=20)
    return ax


def add_north_arrow(ax, xy, yheight=0.05, xwidth=0.04, marg=0.01):
    """
    Add a north arrow to the plot, using relative Axes units

    Parameters
    -----------
    ax : mpl.axis
        current axis to add north arrow
    xy : tuple of 2 floats
        x and y-coordinates of the top point of the north arrow, in relative axes units
    yheigth : float
        distance in map units between the top point and the moddle point of
        the arrow
    xwidth : float
        width of the arrow in map units
    marg : float
        distance between text and rectangles in map units
    """
    from matplotlib.patches import Polygon

    x_a, y_a = xy

    # add north arrow
    ylow = np.sqrt(yheight**2 - xwidth**2)

    # triangles to make north arrow
    rpol = Polygon(np.array([[x_a, y_a], [x_a, y_a - yheight],
                   [x_a + xwidth/2., y_a - yheight - ylow]]),
                   facecolor='w', edgecolor='k', transform=ax.transAxes,
                   zorder=1, clip_on=False)
    ax.add_patch(rpol)
    lpol = Polygon(np.array([[x_a, y_a], [x_a, y_a - yheight],
                   [x_a - xwidth/2., y_a - yheight - ylow]]),
                   facecolor='k', edgecolor='k', transform=ax.transAxes,
                   zorder=1, clip_on=False)
    ax.add_patch(lpol)

    # N text
    ax.text(x_a, y_a + marg, "N", horizontalalignment='center',
            transform=ax.transAxes, zorder=1, size=20)
    return ax

###############
## Load data ##
###############

# Load contact matrices
contacts = pd.read_csv(os.path.join(abs_dir, '../data/interim/epi/contacts/matrices/FR/comesf_formatted_matrices.csv'), index_col=[0,1,2,3,4,5])
contacts = contacts.loc[slice(None), slice(None), type_day, vacation, slice(None), slice(None)][type_contact]

# Compute number of contacts outside of work (doesn't depend on economic activity)
age_classes = contacts.index.get_level_values('age_x').unique().values
baseline_n = 0
for location in ['home', 'leisure_public', 'leisure_private', 'school']:
    baseline_n += np.mean(np.sum(contacts.loc[location, 'A', slice(None), slice(None)].values.reshape(2*[len(age_classes),]), axis=1))

###############################
## Destination-base contacts ##
###############################

gpdf_modified = []
for i, country_suffix in enumerate(countries):
    ## Load data
    # Load geojson file and set index to names of spatial units
    gpdf = gpd.read_file(os.path.join(abs_dir, f'../data/interim/epi/shape/{country_suffix}.json'))
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # Load recurrent mobility matrices (normalized by total and not active population!)
    mob = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/mobility/{country_suffix}/recurrent_mobility_normtotal_{country_suffix}.csv'), index_col=0)
    # Load total population
    pop = pd.read_csv(os.path.join(abs_dir, f'../data/interim/epi/demographic/age_structure_{country_suffix}_2019.csv'), index_col=[0,1]).groupby(by='spatial_unit').sum().reindex(mob.index)
    # Compute mobility-modified population
    workpop = np.matmul(np.transpose(mob), pop).squeeze()
    # Load NACE 21 composition at provincial level
    sectors = pd.read_csv(os.path.join(abs_dir, f'../data/interim/eco/labor_market_composition/sector_structure_by_work_{country_suffix}.csv'), index_col=[0,1])['rel']
    ## desination-based average number of contacts 
    # Compute the number of contacts at work
    work_cont = pop.copy()
    for prov in sectors.index.get_level_values('spatial_unit').unique():
        n = 0
        for sector in sectors.index.get_level_values('economic_activity').unique():
            c = np.mean(np.sum(contacts.loc['work', sector, slice(None), slice(None)].values.reshape(2*[len(age_classes),]), axis=1))
            n += workpop.loc[prov]*sectors.loc[prov, sector]*c
        work_cont.loc[prov] = n 

    # Add baseline contacts and divide by the number of inhabitants to arive at the desination-based average number of contacts
    cont = pop.copy()
    cont.loc[slice(None)] = np.expand_dims(np.squeeze(work_cont.values)/np.squeeze(pop.values) + baseline_n, axis=1)
    # Assign to geopandas
    gpdf['n_contacts_destination'] = 0
    for prov in gpdf.index:
        gpdf.loc[prov, 'n_contacts_destination'] = cont.loc[prov].values
    
    ## Origin-based
    
    # Normalize every column of the mobility matrix so that it adds up to one (Then you know the distribution of where people came from to work in a given spatial patch)
    mob = mob.div(mob.sum(axis=0), axis=1)

    # Use mobility matrix to distribute the total number of contacts in a given spatial patch across the places where the people making those contacts came from
    c = np.zeros(len(gpdf.index))
    for prov in gpdf.index:
        c += np.squeeze(mob[prov].values)*work_cont.loc[prov,:].values
    cont = pop.copy()
    cont[slice(None)] = c

    # Compute per inhabitant
    cont.loc[slice(None)] = np.expand_dims(np.squeeze(cont.values)/np.squeeze(pop.values) + baseline_n, axis=1)

    # Assign to geopandas
    gpdf['n_contacts_origin'] = 0
    for prov in gpdf.index:
        gpdf.loc[prov, 'n_contacts_origin'] = cont.loc[prov].squeeze()

    # Save result
    gpdf_modified.append(gpdf)

row_titles = ['Origin-based number of contacts', 'Destination-based number of contacts']
row_columns = ['n_contacts_origin', 'n_contacts_destination']
props = dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=1)
vmin=16
vmax=24
gpdf_modified_list = gpdf_modified
for row_column in row_columns:
    # Plot data
    fig,axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)
    for (ax,country,sb_position, sb_length, fac, numdiv, lw, gpdf_modified) in zip(axs, countries, sb_positions, sb_lengths, facs, numdivs, lws, gpdf_modified_list):
        # boundaries
        gpdf_modified.boundary.plot(ax=ax, color='black', linewidth=0.8, alpha=1)
        # scalebar
        ax = add_scalebar(ax, sb_position, length=sb_length,
                        fac=fac, numdiv=numdiv, lw=lw)
        # bounding box
        ax.axis('off')
        # data
        if country == 'BE':
            from mpl_toolkits.axes_grid1 import make_axes_locatable
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.4)
            gpdf_modified.plot(column=row_column, legend=True, ax=ax, cmap='Blues', vmin=vmin, vmax=vmax, cax=cax,
                                legend_kwds={"label": "Total number of social contacts (-)"})
        else:
            gpdf_modified.plot(column=row_column, ax=ax, cmap='Blues', vmin=vmin, vmax=vmax)

    # Adjust the position of the right subplot to align it at the bottom
    ax.set_position([0.5, 0.125, 0.5, 0.5])  # [left, bottom, width, height]
    # scale and north arrow
    ax = add_north_arrow(ax, na_position, yheight=na_yheight, xwidth=na_xwidth,)

    # place a text box in upper left in axes coords
    #ax.text(0, 1.55, 'Origin-based number of contacts', transform=ax.transAxes, fontsize=11,
    #        verticalalignment='top', bbox=props)
    #ax.text(0, 1.55, 'Destination-based number of contacts', transform=ax.transAxes, fontsize=11,
    #        verticalalignment='top', bbox=props)

    #plt.tight_layout()
    plt.savefig(f'{row_column}.pdf')
    plt.show()
    plt.close()