#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-10 17:05:16
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-12 02:05:23

import hw2
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle


class pb4(hw2.hw2):

    def qa(self):
        xx = np.mat(np.linspace(-5, 5, 100)).transpose()
        n = len(xx)

        for lb, sigma in zip(range(3), [0.3, 0.5, 1.0]):
            invsig = 1. / (2 * sigma**2)
            mm = np.zeros(n)
            kk = np.mat(np.zeros([n, n]))
            for i in range(n):
                for j in range(n):
                    kk[i, j] = np.exp(-(xx[i] - xx[j])**2 * invsig)

            self.set_111plt((10, 7))

            for j in range(6):
                yy = np.random.multivariate_normal(mm, kk)
                self.ax.plot(xx, yy, '-o', lw=4, color=self.colorlist[j])

            self.add_x_labels(cycle(["x"]), self.ax)
            self.add_y_labels(cycle(["y(x)"]), self.ax)
            self.fig.savefig("figQ4a{:03d}.png".format(
                lb, sigma), **self.figsave)

    def qb(self):
        for lb, sigma in zip(range(3), [0.3, 0.5, 1.0]):
            invsig = 1. / (2 * sigma**2)

            xu = np.mat(np.linspace(-5, 5, 100)).transpose()
            n = len(xu)
            yu = np.mat(np.zeros(n)).transpose()
            print yu.shape
            # mu = np.zeros()

            kuu = np.mat(np.zeros([n, n]))
            for i in range(n):
                for j in range(n):
                    kuu[i, j] = np.exp(-(xu[i] - xu[j])**2 * invsig)

            # (−1.3,2),(2.4,5.2),(−2.5,−1.5),(−3.3,−0.8),(0.3,0.3)
            xd = np.mat([-1.3, 2.4, -2.5, -3.3, 0.3]).transpose()
            yd = np.mat([2, 5.2, -1.5, -0.8, 0.3]).transpose()
            m = len(xd)
            md = np.mat(np.zeros(yd.shape))

            kud = np.mat(np.zeros([n, m]))
            for i in range(n):
                for j in range(m):
                    kud[i, j] = np.exp(-(xu[i] - xd[j])**2 * invsig)

            kdu = np.mat(np.zeros([m, n]))
            for i in range(m):
                for j in range(n):
                    kdu[i, j] = np.exp(-(xd[i] - xu[j])**2 * invsig)

            kdd = np.mat(np.zeros([m, m]))
            for i in range(m):
                for j in range(m):
                    kdd[i, j] = np.exp(-(xd[i] - xd[j])**2 * invsig)

            mstar = kud * np.linalg.inv(kdd) * (yd - md)
            kstar = kuu - kud * np.linalg.inv(kdd) * kdu

            self.set_111plt((10, 7))
            for j in range(5):
                yu = np.random.multivariate_normal(mstar.A1, kstar)
                self.ax.plot(xu, yu, '->', lw=4, color=self.colorlist[j])
            self.ax.plot(xu, mstar, '-o', lw=8, label="mean",
                         color=self.colorlist[5])
            self.ax.plot(xd, yd, '*', label="data", markersize=10, color='k')
            self.add_x_labels(cycle(["x"]), self.ax)
            self.add_y_labels(cycle(["y(x)"]), self.ax)
            self.add_legends(self.ax)
            self.set_tick_size(self.ax)
            self.fig.savefig("figQ4b{:03d}.png".format(
                lb, sigma), **self.figsave)


if __name__ == '__main__':
    drv = pb4()
    drv.qa()
    drv.qb()
