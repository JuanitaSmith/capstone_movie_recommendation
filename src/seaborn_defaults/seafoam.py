import matplotlib.pyplot as plt
import seaborn as sns


class Seafoam:
    """ Default settings for seaborn plots

        This class set default colors inspired by the ocean.
        Fonts for titles, x and y labels are standarized.
    """

    def __init__(self):
        # custom color palette - called ocean spray
        self.hex_colors = [
            '7C9E9E',  # base
            'E2E9E9',  # light gray
            '578686',  # group 1
            '667595',  # group 2
            'FFEDC8',  # highlight
            '366F6F',  # highlight intense
            'DFC591',  # base_complementary
        ]

        # initialize default color parameters
        self.base_color = None
        self.base_grey = None
        self.base_color_group1 = None
        self.base_color_group2 = None
        self.base_highlight = None
        self.base_highlight_intense = None
        self.base_complimentary = None
        self.symbols = None

        # convert hex colors to rgb colors seaborn can understand
        self.rgb_colors = list(map(self.hex_to_rgb, self.hex_colors))
        self.set_plot_defaults()

    def hex_to_rgb(self, hex_value):
        """ Convert color hex code to rgb for sns mapping """

        h = hex_value.lstrip('#')
        return tuple(int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4))

    def set_plot_defaults(self):
        """Set defaults formatting for consistency across all plots """

        # set plot style
        sns.set_style("white")

        self.base_color = self.rgb_colors[0]
        self.base_grey = self.rgb_colors[1]
        self.base_color_group1 = self.rgb_colors[2]
        self.base_color_group2 = self.rgb_colors[3]
        self.base_highlight = self.rgb_colors[4]
        self.base_highlight_intense = self.rgb_colors[5]
        self.base_complimentary = self.rgb_colors[6]

        # up and down arrows for growth indicators
        self.symbols = [u'\u25BC', u'\u25B2']

        small_size = 8
        medium_size = 10
        bigger_size = 12

        # controls default text and font sizes
        plt.rc('font', size=small_size, weight='ultralight', family='sans-serif')
        plt.rc('axes', titlesize=bigger_size, titlecolor='black', titleweight='bold', labelsize=medium_size,
               labelcolor='black', labelweight='ultralight')  # axes settings
        # fontsize of the ytick labels
        plt.rc('xtick', labelsize=small_size)
        # fontsize of the xtick labels
        plt.rc('ytick', labelsize=small_size)
        # legend fontsize
        plt.rc('legend', fontsize=small_size)
        # fontsize of the figure title
        plt.rc('figure', titlesize=bigger_size, titleweight="bold", figsize=[8, 4])

        return (self.base_color, self.base_highlight_intense, self.base_highlight, self.base_complimentary,
                self.base_grey, self.base_color_group1, self.base_color_group2, self.symbols)
