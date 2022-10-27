import os
import numpy as np

from matplotlib.pyplot import figure

from zarth_utils.general_utils import get_random_time_stamp, makedir_if_not_exist

dir_figures = os.path.join(os.getcwd(), "figures")
makedir_if_not_exist(dir_figures)


class Drawer:
    def __init__(self, num_row=1, num_col=1, unit_length=10):
        """
        Init the drawer with the (width=num_col*unit_length, height=num_row*unit_length).
        :param num_row: the number of rows
        :type num_row: int
        :param num_col: the number of columns
        :type num_col: int
        :param unit_length: the length of unit
        :type unit_length: float
        """
        self.num_row = num_row
        self.num_col = num_col
        self.figure = figure(figsize=(num_col * unit_length, num_row * unit_length))

    def draw_one_axes(self, x, y, labels=None, *, index=1, nrows=None, ncols=None,
                      title="", xlabel="", ylabel="", use_marker=False, linewidth=5,
                      fontsize=15, xlim=None, ylim=None):
        """
        Draw one axes, which can be understood as a sub-figure.
        :param x: the data for x axis, list
        :param y: the data for y axis, list of line lists. e.g. [[1, 2, 3], [2, 3, 1]], list
        :param labels: the list of labels of each line, list
        :param index: The subplot will take the index position on a grid with nrows rows and ncols columns.
        :type index: int
        :param nrows: the number of rows in the figure
        :type nrows: int
        :param ncols: the number of columns in the figure
        :type ncols: int
        :param title: the title of the axes
        :type title: str
        :param xlabel: the label for x axis
        :type xlabel: str
        :param ylabel: the label for x axis
        :type ylabel: str
        :param use_marker: whether use markers to mark the points, default=False
        :type use_marker: bool
        :param linewidth: the width of the lines
        :param fontsize: the size of the fonts
        :param xlim: the range of x axis
        :param ylim: the range of y axis
        :return:
        :rtype:
        """
        nrows = self.num_row if nrows is None else nrows
        ncols = self.num_col if ncols is None else ncols

        ax = self.figure.add_subplot(nrows, ncols, index)

        format_generator = self.get_format(use_marker)
        for i, yi in enumerate(y):
            if len(x) == len(y):
                xi = x[i]
            elif len(x) == len(y[0]):
                xi = x
            else:
                raise NotImplementedError

            len_no_nan = 0
            while len_no_nan < len(yi) and not (np.isnan(yi[len_no_nan]) or np.isinf(yi[len_no_nan])):
                len_no_nan += 1
            if len_no_nan == 0:
                continue

            fmt = next(format_generator)

            if labels is not None:
                ax.plot(xi[:len_no_nan], yi[:len_no_nan], fmt, label=labels[i], linewidth=linewidth)
            else:
                ax.plot(xi[:len_no_nan], yi[:len_no_nan], fmt, linewidth=linewidth)

        ax.set_xlabel(xlabel, fontsize=fontsize)
        ax.set_ylabel(ylabel, fontsize=fontsize)
        ax.set_title(title, fontsize=fontsize)
        ax.xaxis.set_tick_params(labelsize=fontsize)
        ax.yaxis.set_tick_params(labelsize=fontsize)
        if xlim is not None:
            ax.set_xlim(*xlim)
        if ylim is not None:
            ax.set_ylim(*ylim)
        if labels is not None:
            ax.legend(fontsize=fontsize)

    def show(self):
        """
        To show the figure.
        """
        self.figure.show()

    def save(self, fname=None):
        """
        To save the figure as fname.
        :param fname: the filename
        :type fname: str
        """
        if fname is None:
            fname = get_random_time_stamp()
        fname = "%s.jpeg" % fname if not fname.endswith(".config") else fname
        self.figure.savefig(os.path.join(dir_figures, fname))

    def clear(self):
        """
        Clear the figure.
        """
        self.figure.clf()

    @staticmethod
    def get_format(use_marker=False):
        """
        Get the format of a line.
        :param use_marker: whether use markers for points or not.
        :type use_marker: bool
        """
        p_color, p_style, p_marker = 0, 0, 0
        colors = ['r', 'g', 'b', 'c', 'm', 'y', 'k']
        styles = ['-', '--', '-.', ':']
        markers = [""]
        if use_marker:
            markers = ['o', 'v', '^', '<', '>', '1', '2', '3', '4', '8', 's', 'p', 'P', '*', 'h', 'H', '+',
                       'x', 'X', 'D', 'd', '|', '_', ]

        while True:
            yield colors[p_color] + styles[p_style] + markers[p_marker]
            p_color += 1
            p_style += 1
            p_marker += 1
            p_color %= len(colors)
            p_style %= len(styles)
            p_marker %= len(markers)
