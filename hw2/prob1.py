#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-09 14:15:27
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-09 23:45:14

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

    def qa(self):
        xx, yy = self.xtrain, self.ytrain
        phi = self.get_feature(xx, 1)
        (mse, ww, yt) = self.closedformMSE(phi, yy)

        xx, yy = self.xtest, self.ytest
        phi = self.get_feature(xx, 1)
        yt = phi * ww
        mse = np.real(np.mean(np.power(yt - yy, 2)))
        stdout.write("rmse = {}\n".format(mse))
        self.ploterr(xx, yy, yt, "figQ1a.png")

    def qb(self):
        xtrain, ytrain = self.xtrain, self.ytrain
        sigma_paras = [0.1, 0.2, 0.4, 0.8, 1.6]  # bandwidth parameters
        n = xtrain.shape[0]
        phixt = self.get_feature(xtrain, 1)
        # sgm = 0.2
        sgm = 0.2
        yt = np.zeros(ytrain.shape)
        for i in range(n):  # calculate R matrix for each x[i]
            rr = np.exp(-0.5 * (np.power(xtrain[i] - xtrain, 2)) / sgm**2)
            sr = np.diag(np.sqrt(rr).A1)
            ww = np.linalg.pinv(sr.transpose() * phixt) * sr * ytrain
            yt[i] = phixt[i] * ww

        xtest, ytest = self.xtest, self.ytest
        n = xtest.shape[0]
        yt = np.zeros(ytest.shape)
        for i in range(n):
            rr = np.exp(-0.5 * (np.power(xtest[i] - xtrain, 2)) / sgm**2)
            sr = np.diag(np.sqrt(rr).A1)
            ww = np.linalg.pinv(sr.transpose() * phixt) * sr * ytrain
            yt[i] = phixt[i] * ww
        self.ploterr(xtest, self.ytest, yt, "figQ1b1.png")
        # np.savetxt(stdout, r0, fmt="%.4f")

    def closedformMSE(self, phi, yy):
        ww = np.linalg.pinv(phi) * yy
        yt = phi * ww
        rmse = np.real(np.mean(np.power(yt - yy, 2)))
        return rmse, ww, yt

    def ploterr(self, xx, yy, yt, fnm):
        self.set_111plt()
        self.ax.plot(xx, yy, 'o', label='labels')
        # self.ax.plot(xx, yt, '<', label='linear')
        self.ax.plot(xx, yt, '<', label='locally weighted linear regression')
        self.add_x_labels(cycle(["epochs"]), self.ax)
        self.add_y_labels(cycle(["mse"]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig(fnm, **self.figsave)


if __name__ == '__main__':
    drv = pb1()
    drv.qa()
    drv.qb()
