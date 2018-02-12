#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-09 14:15:27
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-12 14:33:41

import hw2
import numpy as np
from sys import stdout
from itertools import cycle


def data_generator(size, noise_scale=0.05):
    xs = np.random.uniform(low=0, high=3, size=size)
    # for function y = 0.5x - 0.3 + sin(x) + epsilon, where epsilon is a
    # gaussian noise with std dev= 0.05
    ys = xs * 0.5 - 0.3 + \
        np.sin(3 * xs) + np.random.normal(loc=0, scale=noise_scale, size=size)
    return np.mat(xs), np.mat(ys)


class pb1(hw2.hw2):

    def __init__(self):
        hw2.hw2.__init__(self)
        self.loaddata()
        self.normalize_train_and_test()

    def loaddata(self):
        noise_scales = [0.05, 0.2]
        noise_scale = noise_scales[0]  # choose the first kind of noise scale

        # generate the data form generator given noise scale
        self.xtrain, self.ytrain = data_generator(
            (100, 1), noise_scale=noise_scale)
        self.xtest, self.ytest = data_generator(
            (30, 1), noise_scale=noise_scale)

        self.xtrain = np.mat(self.xtrain)
        self.ytrain = np.mat(self.ytrain)
        self.xtest = np.mat(self.xtest)
        self.ytest = np.mat(self.ytest)

    def qa(self):
        xtrain, ytrain = self.xtrain, self.ytrain
        xtest, ytest = self.xtest, self.ytest
        phitrain = self.get_feature(xtrain, 1)

        (mse, ww, ptrain) = self.closedformMSE(phitrain, ytrain)
        phitest = self.get_feature(xtest, 1)
        ptest = phitest * ww
        mse = np.real(np.mean(np.power(ptest - ytest, 2)))
        stdout.write("test mse = {}\n".format(mse))
        self.plotlabels(xtest, ytest, ptest, [
            'test', 'linear regression'], "figQ1a.png")

    def qb(self):
        xtrain, ytrain = self.xtrain, self.ytrain
        xtest, ytest = self.xtest, self.ytest
        sigma_paras = [0.1, 0.2, 0.4, 0.8, 1.6, 2.0]  # bandwidth parameters

        n = xtrain.shape[0]
        phitrain = self.get_feature(xtrain, 1)
        phitest = self.get_feature(xtest, 1)
        rcd = np.ndarray([len(sigma_paras), 2])

        for it, sgm in zip(range(len(sigma_paras)), sigma_paras):
            n = xtest.shape[0]
            ptest = np.zeros(ytest.shape)
            for i in range(n):
                rr = np.exp(-0.5 * (np.power(xtest[i] - xtrain, 2)) / sgm**2)
                rmat = np.diag(rr.A1)
                ww = np.linalg.inv(phitrain.transpose() *
                                   rmat * phitrain) * phitrain.transpose() * rmat * ytrain
                ptest[i] = phitest[i] * ww

            if sgm in [0.2, 2.0]:
                self.plotlabels(xtest, ytest, ptest,
                                ["test", r"LWR $\tau$ = {:.3f}".format(sgm)], "figQ1b{:03}.png".format(it))

            mse = np.real(np.mean(np.power(ptest - ytest, 2)))
            stdout.write("bandwidth = {:.3f}, mse = {:.3f}\n".format(sgm, mse))
            rcd[it, 0], rcd[it, 1] = sgm, mse
        self.ploterr(rcd, "figQ1b3.png")

    def closedformMSE(self, phi, yy):
        ww = np.linalg.pinv(phi) * yy
        yt = phi * ww
        mse = np.real(np.mean(np.power(yt - yy, 2)))
        return mse, ww, yt

    def plotlabels(self, xx, yy, yt, lb, fnm):
        self.set_111plt((10, 7))
        self.ax.plot(xx, yy, 'o', label=lb[0], markersize=12)
        self.ax.plot(xx, yt, '<', label=lb[1], markersize=12)
        self.add_x_labels(cycle(["x"]), self.ax)
        self.add_y_labels(cycle(["y"]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig(fnm, **self.figsave)

    def ploterr(self, rcd, fnm):
        self.set_111plt((10, 7))
        self.ax.plot(rcd[:, 0], rcd[:, 1], '-o',
                     label="MSE vs. Kernerl Width", markersize=12, color=self.colorlist[1])
        self.add_x_labels(cycle(["Kernel width"]), self.ax)
        self.add_y_labels(cycle(["MSE"]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig(fnm, **self.figsave)

if __name__ == '__main__':
    drv = pb1()
    drv.qa()
    drv.qb()
