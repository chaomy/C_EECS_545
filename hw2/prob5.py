#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-11 11:19:34
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-11 23:34:04

import hw2
import numpy as np
from itertools import cycle


class pb5(hw2.hw2):

    def loaddata(self):
        data = np.zeros((100, 3))
        val = np.random.uniform(0, 2, 100)
        diff = np.random.uniform(-1, 1, 100)
        data[:, 0], data[:, 1], data[:, 2] = val - \
            diff, val + diff, np.ones(100)
        target = np.asarray(val > 1, dtype=int) * 2 - 1
        return data, target

    def loaddatab(self):
        data = np.ones((100, 3))
        data[:50, 0], data[50:, 0] = np.random.normal(
            0, 1, 50), np.random.normal(2, 1, 50)
        data[:50, 1], data[50:, 1] = np.random.normal(
            0, 1, 50), np.random.normal(2, 1, 50)
        target = np.zeros(100)
        target[:50], target[50:] = -1 * np.ones(50), np.ones(50)
        return data, target

    def qa(self):
        data, target = self.loaddata()
        ww, acc = self.perceptron(np.mat(data), target)
        self.plt(data, target, ww, acc, "figQ5a.png", [2, 4, 3])
        return

    def qb(self):
        data, target = self.loaddatab()
        ww, acc = self.perceptron(np.mat(data), target)
        self.plt(data, target, ww, acc, "figQ5b1.png", [2, 4, 3])

        ww, acc = self.votedperceptron(np.mat(data), target)
        self.plt(data, target, ww, acc, "figQ5b2.png", [2, 4, 3])
        return

    def perceptron(self, xx, yy):
        (n, m) = xx.shape
        ww = np.mat(np.zeros(m)).transpose()
        # w = wn or wn + tn * phi(xn)
        tt = -1 * np.ones(len(yy))
        epochs = 10
        acc = np.zeros(epochs)
        for j in range(10):
            for i in range(n):
                tm = yy[i] * xx[i] * ww
                if tm <= 0:
                    ww += yy[i] * xx[i].transpose()
            tt[np.where(xx * ww > 0)[0]] = 1
            acc[j] = len(np.where(tt == yy)[0])
            tt = -1 * np.ones(len(yy))
        return ww, acc[-1] / n

    def votedperceptron(self, xx, yy):
        (n, m) = xx.shape
        ww = np.mat(np.zeros(m)).transpose()
        # w = wn or wn + tn * phi(xn)
        tt = -1 * np.ones(len(yy))
        epochs = 20
        acc = np.zeros(epochs)
        wws = np.ndarray([epochs, m])

        for j in range(epochs):
            for i in range(n):
                tm = yy[i] * xx[i] * ww
                if tm <= 0:
                    ww += yy[i] * xx[i].transpose()
            tt[np.where(xx * ww > 0)[0]] = 1
            acc[j] = len(np.where(tt == yy)[0])
            wws[j, :] = ww.A1
            tt = -1 * np.ones(len(yy))
        return np.mat(wws[np.argmax(acc)]).transpose(), np.max(acc) / n

    def plt(self, data, target, ww, acc, fnm="figQ5.png", cl=[0, 1, 2]):
        ia = np.where(target > 0)
        ib = np.where(target < 0)

        self.set_111plt((10, 7))
        self.ax.scatter(data[:, 0][ia], data[:, 1][ia],
                        color=self.colorlist[cl[0]])
        self.ax.scatter(data[:, 0][ib], data[:, 1][ib],
                        color=self.colorlist[cl[1]])

        # wx = 0
        xx = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
        yy = (-ww[2, 0] - ww[0, 0] * xx) / ww[1, 0]
        self.ax.plot(xx, yy, label="accuracy = {:.2f}".format(acc),
                     lw=5, color=self.colorlist[cl[2]])
        self.add_x_labels(cycle(["x"]), self.ax)
        self.add_y_labels(cycle(["y"]), self.ax)
        self.add_legends(self.ax)
        self.ax.set_xlim(np.min(data[:, 0]), np.max(data[:, 0]))
        self.ax.set_ylim(np.min(data[:, 1]), np.max(data[:, 1]))
        self.fig.savefig(fnm, **self.figsave)


if __name__ == '__main__':
    drv = pb5()
    drv.qa()
    drv.qb()
