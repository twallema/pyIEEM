import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

cmap = {"orange": "#E69F00", "light_blue": "#56B4E9",
        "green": "#009E73", "yellow": "#F0E442",
        "blue": "#0072B2", "red": "#D55E00",
        "pink": "#CC79A7", "black": "#000000"}

abs_dir = os.path.dirname(__file__)

# settings
countries = ['SWE', 'BE']
sectors = ['A', 'B', 'C']
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

##############################
## labor market composition ##
##############################

fig, axs = plt.subplots(nrows=1, ncols=len(countries),
                        figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})
# loop over countries
for (ax, country, sb_position, sb_length, fac, numdiv, lw) in zip(axs, countries, sb_positions, sb_lengths, facs, numdivs, lws):

    # load geospatial data

    # Load geojson file
    gpdf = gpd.read_file(os.path.join(
        abs_dir, f'../data/interim/epi/shape/{country}.json'))
    # set index to names of spatial units
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # sort alphabetically
    gpdf = gpdf.sort_index()

    # compute fraction of inhabitants working in sectors `sectors`
    # load labor market composition
    lmc = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/eco/labor_market_composition/sector_structure_by_work_{country}.csv'), index_col=[0, 1]).sort_index()['rel']
    # sum over `sectors`
    lmc = 100*lmc.loc[slice(None), sectors].groupby(by='spatial_unit').sum()
    # assign to gdpf
    gpdf['lmc'] = lmc.values
    # load demography per spatial patch
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().sort_index().squeeze()
    # compute demographic mean
    # print(sum(demography*lmc/sum(demography)))
    # print(lmc)
    # plot a nice map
    # boundaries
    gpdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    # add scalebar
    ax = add_scalebar(ax, sb_position, length=sb_length,
                      fac=fac, numdiv=numdiv, lw=lw)
    # legend
    ax.axis('off')

    # plot data
    # population density
    if country == 'BE':
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        gpdf.plot(ax=ax, column='lmc', vmin=5, vmax=20, cmap='Blues', legend=True, cax=cax,
                  legend_kwds={"label": "Fraction employed in sectors A, B, and C (%)"})
    else:
        gpdf.plot(ax=ax, column='lmc', vmin=5, vmax=20, cmap='Blues')

# Adjust the position of the right subplot to align it at the bottom
ax.set_position([0.5, 0.125, 0.5, 0.5])  # [left, bottom, width, height]

# scale and north arrow
ax = add_north_arrow(
    ax, na_position, yheight=na_yheight, xwidth=na_xwidth,)

plt.savefig('map_lmc_ABC.pdf', dpi=300)
# plt.show()
plt.close()

############################################
## population density + outbound mobility ##
############################################

fig, axs = plt.subplots(nrows=1, ncols=len(countries),
                        figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})
# loop over countries
for (ax, country, sb_position, sb_length, fac, numdiv, lw) in zip(axs, countries, sb_positions, sb_lengths, facs, numdivs, lws):

    # load geospatial data

    # Load geojson file
    gpdf = gpd.read_file(os.path.join(
        abs_dir, f'../data/interim/epi/shape/{country}.json'))
    # set index to names of spatial units
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # sort alphabetically
    gpdf = gpdf.sort_index()

    # compute fraction of inhabitants crossing NUTS2 border during their commute
    # load mobility
    mob = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/mobility/{country}/recurrent_mobility_normactive_{country}.csv'), index_col=0).sort_index(axis=0).sort_index(axis=1)
    # load demography per spatial patch
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().sort_index().squeeze()

    # extract relative fraction of travellers per spatial patch
    travellers = []
    for i, spatial_unit in enumerate(mob.columns):
        travellers.append(
            np.sum(mob.loc[spatial_unit]) - mob.loc[spatial_unit][i])
    travellers = np.array(travellers)

    # print(100*np.sum(mob,axis=1))
    # print(100*sum(np.sum(mob,axis=1)*(demography/sum(demography))))
    # print(100*travellers)
    # print(100*sum(travellers*(demography/sum(demography))))

    # plot a nice map
    # boundaries
    gpdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    # add scalebar
    ax = add_scalebar(ax, sb_position, length=sb_length,
                      fac=fac, numdiv=numdiv, lw=lw)
    # legend
    ax.axis('off')

    # plot data
    # population density
    gpdf.pop_density = np.log10(gpdf.pop_density.values)
    if country == 'BE':
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        gpdf.plot(ax=ax, column='pop_density', vmin=0, vmax=3.1, cmap='Blues', legend=True, cax=cax,
                  legend_kwds={"label": "Log. population density (inhab. km$^{-2}$)", "ticks": [0, 1, 2, 3]})
    else:
        gpdf.plot(ax=ax, column='pop_density', vmin=0, vmax=3.1, cmap='Blues')

    # mobility
    c_x = []
    c_y = []
    for coord in gpdf.centroid:
        c_x.append(coord.x)
        c_y.append(coord.y)
    ax.scatter(c_x, c_y, s=800*travellers, marker="o",
               edgecolors='black', facecolors='red', alpha=0.8, zorder=1)

# Adjust the position of the right subplot to align it at the bottom
ax.set_position([0.5, 0.125, 0.5, 0.5])  # [left, bottom, width, height]

# scale and north arrow
ax = add_north_arrow(
    ax, na_position, yheight=na_yheight, xwidth=na_xwidth,)

plt.savefig('map_popdens.pdf', dpi=300)
# plt.show()
plt.close()

################
## employment ##
################

fig, axs = plt.subplots(nrows=1, ncols=len(countries),
                        figsize=figsize, gridspec_kw={'width_ratios': [1, 1]})
# loop over countries
for (ax, country, sb_position, sb_length, fac, numdiv, lw) in zip(axs, countries, sb_positions, sb_lengths, facs, numdivs, lws):

    # load geospatial data

    # Load geojson file
    gpdf = gpd.read_file(os.path.join(
        abs_dir, f'../data/interim/epi/shape/{country}.json'))
    # set index to names of spatial units
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # sort alphabetically
    gpdf = gpdf.sort_index()

    # compute fraction of inhabitants crossing NUTS2 border during their commute
    # load mobility
    mob = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/mobility/{country}/recurrent_mobility_normactive_{country}.csv'), index_col=0).sort_index(axis=0).sort_index(axis=1)
    # load demography per spatial patch
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().sort_index().squeeze()
    # compute employed fraction per spatial patch
    employment = mob.sum(axis=1)
    # print relevant statistics
    print(sum((employment*demography)/sum(demography)), employment)
    # append to geopandas dataframe
    gpdf['fraction_employed'] = 100*employment

    # plot a nice map
    # boundaries
    gpdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    # add scalebar
    ax = add_scalebar(ax, sb_position, length=sb_length,
                      fac=fac, numdiv=numdiv, lw=lw)
    # legend
    ax.axis('off')

    # plot data
    # employment
    if country == 'BE':
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.4)
        gpdf.plot(ax=ax, column='fraction_employed', vmin=55, vmax=85, cmap='Blues', legend=True, cax=cax,
                  legend_kwds={"label": "Employed fraction (%)", "ticks": [60, 70, 80]})
    else:
        gpdf.plot(ax=ax, column='fraction_employed',
                  vmin=55, vmax=85, cmap='Blues')

# Adjust the position of the right subplot to align it at the bottom
ax.set_position([0.5, 0.125, 0.5, 0.5])  # [left, bottom, width, height]

# scale and north arrow
ax = add_north_arrow(
    ax, na_position, yheight=na_yheight, xwidth=na_xwidth,)

plt.savefig('map_employment.pdf', dpi=300)
# plt.show()
plt.close()

#####################################################################
## relationship popdens and employment, fraction outbound mobility ##
#####################################################################

markers = ['o', '^']
colors = [cmap['red'], 'black']

fig, axs = plt.subplots(nrows=1, ncols=len(countries),
                        figsize=(11.7, 8.3/2), sharey=True)

# loop over countries
for i, (country, color, marker) in enumerate(zip(countries, colors, markers)):
    # load geospatial data
    # Load geojson file
    gpdf = gpd.read_file(os.path.join(
        abs_dir, f'../data/interim/epi/shape/{country}.json'))
    # set index to names of spatial units
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # sort alphabetically
    gpdf = gpdf.sort_index()
    # compute fraction of inhabitants crossing NUTS2 border during their commute
    # load mobility
    mob = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/mobility/{country}/recurrent_mobility_normactive_{country}.csv'), index_col=0).sort_index(axis=0).sort_index(axis=1)
    # load demography per spatial patch
    demography = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/demographic/age_structure_{country}_2019.csv'), index_col=[0, 1]).groupby(by='spatial_unit').sum().sort_index().squeeze()
    # compute employed fraction per spatial patch
    employment = 100*mob.sum(axis=1)
    # extract relative fraction of travellers per spatial patch
    travellers = []
    for i, spatial_unit in enumerate(mob.columns):
        travellers.append(
            np.sum(mob.loc[spatial_unit]) - mob.loc[spatial_unit][i])
    travellers = 100*np.array(travellers)

    # visualise
    axs[0].scatter(np.log10(gpdf.pop_density), travellers,
                   marker=marker, color=color, s=90, alpha=0.9)
    axs[1].scatter(employment, travellers, marker=marker,
                   color=color, label=country, s=90, alpha=0.9)
    # labels
    axs[0].set_ylim([0, 21])
    axs[0].set_ylabel('Outbound commuters (%)')
    axs[0].set_xlabel('Log. population density (inhab.km$^{-2}$)')
    axs[1].set_xlabel('Employed fraction (%)')
    axs[1].legend(framealpha=1)

plt.tight_layout()
plt.savefig('correlations_travellers.pdf', dpi=300)
# plt.show()
plt.close()
