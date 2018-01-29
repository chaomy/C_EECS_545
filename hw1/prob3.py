#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-25 22:15:39
# @Last Modified by:   chaomy
# @Last Modified time: 2018-01-29 00:52:42


from itertools import cycle
import numpy as np
import hw1


class pb3(hw1.hw1):

    def __init__(self):
        hw1.hw1.__init__(self)
        self.normalize_train_and_test()

    def train(self, phi, yy, order=1, lmd=0.1):
        (m, n) = phi.shape
        ww = np.linalg.inv(phi.transpose() * phi +
                           np.identity(n) * lmd * m) * phi.transpose() * yy
        rmse = np.sqrt(np.mean(np.power(phi * ww - yy, 2)))
        return rmse, ww

    def ploterr(self, dat, fnm, xlb, ylb):
        self.set_111plt()
        self.ax.plot(dat[:, 0], dat[:, 1],
                     label="train", **next(self.keysiter))
        self.ax.plot(dat[:, 0], dat[:, 2],
                     label="validation", **next(self.keysiter))
        self.ax.plot(dat[:, 0], dat[:, 3],
                     label="test", **next(self.keysiter))
        self.add_x_labels(cycle([xlb]), self.ax)
        self.add_y_labels(cycle([ylb]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig(fnm, **self.figsave)

    def qb(self):
        xx, yy = self.get_train_data()
        xtest, ytest = self.get_test_data()
        m = int(0.9 * xx.shape[0])  # training set 90%
        tx, ty = xx[:m], yy[:m]
        vx, vy = xx[m:], yy[m:]

        phitx = self.get_feature(tx, 1)
        phivx = self.get_feature(vx, 1)
        phitest = self.get_feature(xtest, 1)

        lmds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        dat = np.ndarray([len(lmds), 4])

        for i, lmd in zip(range(len(lmds)), lmds):
            te, ww = self.train(phitx, ty, 1, lmd)
            dat[i, :] = lmd, te, np.sqrt(np.mean(np.power(phivx * ww - vy, 2))), np.sqrt(np.mean(np.power(phitest * ww - ytest, 2)))

        self.ploterr(dat, "figQ3b.png", r"$\lambda$", "error")
        print dat


if __name__ == '__main__':
    drv = pb3()
    drv.qb()
