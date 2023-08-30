import os
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt

abs_dir = os.path.dirname(__file__)

# settings
countries = ['SWE', 'BE']
# number on map
color = 'red'
textsizes = [20, 30]
# size figure
figsizes = [(8.3, 11.7), (11.7, 8.3)]
# north arrow
na_positions = [(1.10, 0.25), (0.215, 0.31)]
na_xwidths = [0.115, 0.08]
na_yheights = [0.12, 0.11]
# scale
sb_positions = [(2500000, 7400000), (280000, 6370000)]
sb_lengths = [5*165000, 165000]
facs = ['   500 km', '   100 km']
numdivs = [5, 10]
# annotation arrows
arrow_args = dict(arrowstyle="-", lw=2,)

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

####################
## visualisations ##
####################


# loop over countries
for (country, textsize, figsize, na_position, na_xwidth, na_yheight, sb_position, sb_length, fac, numdiv) in zip(countries, textsizes, figsizes, na_positions, na_xwidths, na_yheights, sb_positions, sb_lengths, facs, numdivs):
    # Load geojson file
    gpdf = gpd.read_file(os.path.join(
        abs_dir, f'../data/interim/epi/shape/{country}.json'))
    # set index to names of spatial units
    gpdf.set_index('spatial_unit', inplace=True)
    gpdf.index.name = 'spatial_unit'
    # sort alphabetically
    gpdf = gpdf.sort_index()
    # Load recurrent mobility matrices (normalized by total and not active population!)
    mob = pd.read_csv(os.path.join(
        abs_dir, f'../data/interim/epi/mobility/{country}/recurrent_mobility_normtotal_{country}.csv'), index_col=0)

    ##############
    ## Plot map ##
    ##############

    fig, ax = plt.subplots(figsize=figsize)
    # boundaries
    gpdf.boundary.plot(ax=ax, color='black', linewidth=0.8)
    # number of spatial patch
    if country == 'BE':
        ax.text(515000, 6650000, '1', size=textsize, color=color)  # Antwerpen
        ax.text(495000, 6547000, '2', size=textsize, color=color)  # Brabant Wallon
        # Brussel
        ax.annotate('3', size=textsize, color=color, xy=(0.495, 0.61), xycoords='figure fraction', xytext=(-100, 150), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        ax.text(430000, 6530000, '4', size=textsize, color=color)  # Hainaut
        ax.text(630000, 6530000, '5', size=textsize, color=color)  # Liège
        ax.text(595000, 6610000, '6', size=textsize, color=color)  # Limburg
        ax.text(600000, 6430000, '7', size=textsize, color=color)  # Luxembourg
        ax.text(535000, 6495000, '8', size=textsize, color=color)  # Namur
        ax.text(410000, 6615000, '9', size=textsize, color=color)  # Oost-Vlaanderen
        ax.text(520000, 6593000, '10', size=textsize, color=color)  # Vlaams-Brabant
        ax.text(323000, 6615000, '11', size=textsize, color=color)  # West-Vlaanderen

    else:
        # Blekinge
        ax.annotate('1', size=textsize, color=color, xy=(0.44, 0.187), xycoords='figure fraction', xytext=(0, -100), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Gotland
        ax.annotate('4', size=textsize, color=color, xy=(0.535, 0.235), xycoords='figure fraction', xytext=(40, -40), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Halland
        ax.annotate('5', size=textsize, color=color, xy=(0.39, 0.21), xycoords='figure fraction', xytext=(-60, -60), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Kalmar
        ax.annotate('8', size=textsize, color=color, xy=(0.47, 0.22), xycoords='figure fraction', xytext=(50, -60), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Kronoberg
        ax.annotate('9', size=textsize, color=color, xy=(0.44, 0.205), xycoords='figure fraction', xytext=(50, -90), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Orebro
        ax.annotate('11', size=textsize, color=color, xy=(0.445, 0.325), xycoords='figure fraction', xytext=(-120, -90), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Skane
        ax.annotate('13', size=textsize, color=color, xy=(0.41, 0.17), xycoords='figure fraction', xytext=(-40, -60), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Sodermanland
        ax.annotate('14', size=textsize, color=color, xy=(0.49, 0.31), xycoords='figure fraction', xytext=(70, -40), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Stockholm
        ax.annotate('15', size=textsize, color=color, xy=(0.52, 0.33), xycoords='figure fraction', xytext=(60, -20), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Uppsala
        ax.annotate('16', size=textsize, color=color, xy=(0.515, 0.35), xycoords='figure fraction', xytext=(60, 20), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # Vastmanland
        ax.annotate('20', size=textsize, color=color, xy=(0.475, 0.34), xycoords='figure fraction', xytext=(80, 80), textcoords='offset points', ha="left", va="bottom", arrowprops=arrow_args)
        # others without arrow
        ax.text(1600000, 8550000, '2', size=textsize, color=color)  # 2: Dalarna
        ax.text(1760000, 8700000, '3', size=textsize, color=color)  # 3: Gavleborg
        ax.text(1550000, 9100000, '6', size=textsize, color=color)  # 6: Jamtland
        ax.text(1580000, 7840000, '7', size=textsize, color=color)  # 7: Jonkoping
        ax.text(2100000, 10100000, '10', size=textsize, color=color)  # 10: Norrbotten
        ax.text(1650000, 8010000, '12', size=textsize, color=color)  # 12: Ostergotland
        ax.text(1400000, 8300000, '17', size=textsize, color=color)  # 17: Varmland
        ax.text(1840000, 9550000, '18', size=textsize, color=color)  # 18: Vasterbotten
        ax.text(1840000, 9150000, '19', size=textsize, color=color)  # 19: Vasternorrland
        ax.text(1360000, 7950000, '21', size=textsize, color=color)  # 21: Vastmanland

    # legend
    if country == 'BE':
        textstr = '\n'.join(('Provinces:',
                             '1. Antwerpen',
                             '2. Brabant Wallon',
                             '3. Brussels',
                             '4. Hainaut',
                             '5. Liège',
                             '6. Limburg',
                             '7. Luxembourg',
                             '8. Namur',
                             '9. Oost-Vlaanderen',
                             '10. Vlaams-Brabant',
                             '11. West-Vlaanderen',
                             ))
        props = dict(boxstyle='square', facecolor='none', alpha=1)
        # place a text box in upper left in axes coords
        ax.text(0.98, 0.93, textstr, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', bbox=props)
    else:
        textstr = '\n'.join(('Counties:',
                             '1. Blekinge',
                             '2. Dalarna',
                             '3. Gävleborg',
                             '4. Gotland',
                             '5. Halland',
                             '6. Jämtland',
                             '7. Jönköping',
                             '8. Kalmar',
                             '9. Kronoberg',
                             '10. Norrbotten',
                             '11. Örebro',
                             '12. Östergötland',
                             '13. Skåne',
                             '14. Södermanland',
                             '15. Stockholm',
                             '16. Uppsala',
                             '17. Värmland',
                             '18. Västerbotten',
                             '19. Västernorrland',
                             '20. Västmanland',
                             '21. Västra Götaland'
                             ))
        props = dict(boxstyle='square', facecolor='none', alpha=1)
        # place a text box in upper left in axes coords
        ax.text(1.1, 0.95, textstr, transform=ax.transAxes, fontsize=13,
                verticalalignment='top', bbox=props)

    # scale and north arrow
    ax = add_north_arrow(
        ax, na_position, yheight=na_yheight, xwidth=na_xwidth,)
    ax = add_scalebar(ax, sb_position, length=sb_length,
                      fac=fac, numdiv=numdiv)

    # legend
    ax.axis('off')
    plt.show()
    plt.close()
