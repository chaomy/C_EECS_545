# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-27 14:11:25
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-10 22:29:15

import matplotlib.pyplot as plt
from itertools import cycle

tableau = [
    (255, 187, 120),  # light orange
    (148, 103, 189),  # purple
    (31, 119, 180),  # blue
    (152, 223, 138),  # light green
    (255, 127, 14),  # orange
    (188, 189, 34),  # gloden
    (174, 199, 232),  # light blue
    (44, 160, 44)  # green
]

for i in range(len(tableau)):
    r, g, b = tableau[i]
    tableau[i] = (r / 255., g / 255., b / 255.)


class myplt(object):

    def __init__(self):
        self.myfontsize = 16
        self.mlabelsize = self.myfontsize - 2
        self.mmarkesize = self.myfontsize - 5
        self.mlabelsize = self.myfontsize - 2
        self.mlegensize = self.myfontsize + 1
        self.line = ['--', '-.', '-',
                     '--', '--', '-.', '-']
        self.markers = ['o', '*', 'H', 'p', '3', '2',
                        '4', 'D', 's', 'o', '*', 'H', 'p',
                        '3', '2', '4', 'D', 's']
        self.usemarkers = ['o', '<', 'o', '<', 'o', '<', 'o', '<']
        self.keyslist = []
        self.pltkwargs = {'linestyle': '--', 'linewidth': 4,
                          'markersize': self.mmarkesize}
        self.colorlist = tableau
        self.makeriter = cycle(self.usemarkers)
        self.lineiter = cycle(self.line)
        self.powlim = (-2, 2)
        self.set_color()
        self.set_keys()
        return

    def set_color(self):
        self.tableau = tableau
        self.coloriter = cycle(self.tableau)
        return

    def set_markersize(self, markersize):
        self.mmarkesize = markersize
        self.set_keys()
        return

    def set_phonon_keys(self, loc='best'):
        self.pltkwargs = {"linestyle": '-', "linewidth": 2.0}
        # self.legendarg = {"loc": loc, "fontsize": self.mlegensize}
        self.legendarg = {"loc": loc, "fontsize": self.mlegensize,
                          'mode': 'expand', 'borderaxespad': 0}
        self.figsave = {"bbox_inches": 'tight'}
        return

    def set_keys(self, loc='best', ncol=3, nrow=None, lg='in'):
        self.keyslist = []
        self.pltkwargs = {'linestyle': '--', 'linewidth': 4,
                          'markersize': self.mmarkesize}
        if nrow is not None:
            self.legendarg = {"loc": loc,
                              "fontsize": self.myfontsize,
                              "nrow": nrow,
                              "frameon": True}
        elif lg in ['top', 'out']:
            self.legendarg = {'bbox_to_anchor': (0., 1.00, 1., .100),
                              'loc': 4, 'ncol': ncol, 'mode': "expand",
                              # 'borderaxespad': 0,
                              'fontsize': self.myfontsize}
        else:
            self.legendarg = {"loc": loc,
                              "fontsize": self.myfontsize,
                              "ncol": ncol,
                              "frameon": True}
        self.figsave = {"bbox_inches": 'tight'}
        for i in range(10):
            key = {'color': next(self.coloriter),
                   'marker': next(self.makeriter)}
            key.update(self.pltkwargs)
            self.keyslist.append(key)
        self.keysiter = cycle(self.keyslist)
        return

    def reset_figure(self, isize=(10, 7.0)):
        self.fig = plt.figure(figsize=isize)
        return

    def set_111plt(self, isize=(10., 10.), lim=False):
        self.figsize = isize
        self.fig = plt.figure(figsize=self.figsize)
        self.ax = self.fig.add_subplot(111)
        self.axls = [self.ax]
        if lim is True:
            self.set_pow_lim(self.ax)
        self.axls = [self.ax]
        return

    def set_211plt(self, isize=(11.0, 8.5), lim=True):
        self.fig = plt.figure(figsize=isize)
        self.ax1 = self.fig.add_subplot(211)
        self.ax2 = self.fig.add_subplot(212)
        self.axls = [self.ax1, self.ax2]
        if lim is True:
            self.set_pow_lim(self.ax1, self.ax2)
        self.axls = [self.ax1, self.ax2]
        return

    def set_pow_lim(self, *args):
        for ax in args:
            ax.get_yaxis().get_major_formatter().set_powerlimits(self.powlim)
            ax.get_xaxis().get_major_formatter().set_powerlimits(self.powlim)
        return

    def add_legends(self, *args):
        for ax in args:
            ax.legend(**self.legendarg)
        return

    def add_y_labels(self, labeliter, *args):
        for ax in args:
            ax.set_ylabel(next(labeliter), {'fontsize': self.myfontsize})
        return

    def add_x_labels(self, labeliter, *args):
        for ax in args:
            ax.set_xlabel(next(labeliter), {'fontsize': self.myfontsize})
        return

    def set_tick_size(self, *args, **kwargs):
        for ax in args:
            ax.xaxis.set_tick_params(width=4)
            ax.yaxis.set_tick_params(width=4)
            for tx in ax.xaxis.get_ticklabels():
                tx.set_fontsize(self.mlabelsize)
            for tx in ax.yaxis.get_ticklabels():
                tx.set_fontsize(self.mlabelsize)

            if 'coord' in kwargs.keys():
                print "have coord"
                ax.get_yaxis().set_label_coords(kwargs['coord'], 0.5)
            else:
                ax.get_yaxis().set_label_coords(-0.060, 0.5)
            # ax.get_yaxis().set_label_coords(-0.07, 0.5)
        return

    def add_title(self, mtitle, ax):
        ax.set_title(mtitle, fontsize=self.myfontsize)
        return

    def add_vline(self, xlist, ax):
        color = next(self.coloriter)
        for x in xlist:
            ax.axvline(x, color=color,
                       alpha=0.8, linestyle='--',
                       linewidth=2.)
        return

    def closefig(self):
        plt.close(self.fig)
        return

    def remove_xticks(self, *args):
        for ax in args:
            ax.xaxis.set_major_formatter(plt.NullFormatter())
        return
