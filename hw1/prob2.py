#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-25 22:15:39
# @Last Modified by:   chaomy
# @Last Modified time: 2018-01-29 00:01:47

import hw1
import numpy as np
from itertools import cycle


class pb2(hw1.hw1):

    def __init__(self):
        hw1.hw1.__init__(self)
        self.normalize_train_and_test()

    def train(self, xx, yy, order=1):
        phi = self.get_feature(xx, order)
        ww = np.linalg.pinv(phi) * yy
        rmse = np.sqrt(np.mean(np.power(phi * ww - yy, 2)))
        return rmse, ww

    def test(self, ww, order=1):
        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, order)
        rmse = np.sqrt(np.mean(np.power(phi * ww - yy, 2)))
        return rmse

    def ploterr(self, dat, fnm, xlb, ylb):
        self.set_111plt()
        self.ax.plot(dat[:, 0], dat[:, 1],
                     label="train", **next(self.keysiter))
        self.ax.plot(dat[:, 0], dat[:, 2],
                     label="test", **next(self.keysiter))
        self.add_x_labels(cycle([xlb]), self.ax)
        self.add_y_labels(cycle([ylb]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig(fnm, **self.figsave)

    def qa(self):
        xx, yy = self.get_train_data()
        dat = np.ndarray([5, 3])
        for i in range(5):
            dr, ww = self.train(xx, yy, i)
            tr = self.test(ww, i)
            print("degrees ", i, "train err: ", dr, "test  err: ", tr)
            dat[i, :] = i, dr, tr
        self.ploterr(dat, "figQ2a.png", "order", "error")

    def qb(self):
        xx, yy = self.get_train_data()
        dat = np.ndarray([5, 3])
        for i, rr in zip(range(5), [0.2, 0.4, 0.6, 0.8, 1.0]):
            m = int(rr * xx.shape[0])
            dr, ww = self.train(xx[:m], yy[:m], 1)
            tr = self.test(ww, 1)
            print("amounts ", rr, "train err: ", dr, "test  err: ", tr)
            dat[i, :] = rr, dr, tr
        self.ploterr(dat, "figQ2b.png", "data size", "error")

if __name__ == '__main__':
    drv = pb2()
    drv.qa()
    drv.qb()
