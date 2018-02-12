#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-11 13:50:22
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-12 01:45:36

import hw2
import numpy as np
from sys import stdout
from numpy import log, exp
from itertools import cycle
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split


class pb6(hw2.hw2):

    def loaddata(self):
        # Normalized Binary dataset
        # 4 features, 100 examples, 50 labeled 0 and 50 labeled 1
        X, y = load_breast_cancer().data, load_breast_cancer().target
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.33)
        self.xtrain, self.xtest = np.mat(X_train), np.mat(X_test)
        self.ytrain, self.ytest = np.mat(
            y_train).transpose(), np.mat(y_test).transpose()
        return

    def qa(self):
        self.loaddata()
        self.normalize_train_and_test()
        ww, rcd = self.sgd()
        self.ploterr(rcd)
        self.plotacc(rcd)
        stdout.write("learned parameter vector w \n")
        np.savetxt(stdout, ww.transpose(), fmt="%.4f")
        stdout.write("final training error: {:.4f} test error: {:.4f}\n".format(
            rcd[-1, 1], rcd[-1, 2]))
        stdout.write("final training accuracy: {:.4f} test accuracy: {:.4f}".format(
            rcd[-1, 3], rcd[-1, 4]))
        return

    def sgd(self):
        phi, phitest, ytrain, ytest = self.xtrain, self.xtest, self.ytrain, self.ytest

        (mtrain, ntrain) = phi.shape
        (mtest, ntest) = phitest.shape

        ww = np.mat(-1.0 + 2.0 * np.random.rand(ntrain, 1))
        eta, epochs = 1e-2, 1

        rcd = np.ndarray([mtrain, 5])

        invntrain = 1. / mtrain
        invntest = 1. / mtest

        ix = np.arange(mtrain)
        np.random.shuffle(ix)

        cn = 0
        for i, j in zip(range(len(ix)), ix):
            # loss = tn ln(1 + exp(-w * phi)) + (1 - tn) ln(1 + exp(w * #
            # phi))
            # derivative = phi * (1 / (1 + exp(w * phi)) - t)

            sigma = 1. / (1 + exp(-(phi[j] * ww)[0, 0]))
            gd = phi[j].transpose() * (sigma - ytrain[j][0, 0])

            # updates
            ww = ww - (eta * gd)

            # training error and accuracy
            tm = phi * ww
            trnerr = (ytrain.transpose() * log(1 + exp(-tm)) +
                      (1 - ytrain).transpose() * log(1 + exp(tm))) * invntrain

            tt = np.mat(np.zeros(mtrain)).transpose()
            tt[np.where(tm > 0)[0], 0] = 1
            trnacc = len(np.where(tt == ytrain)[0]) * invntrain

            # calculate test error and accuracy
            tm = phitest * ww
            tsterr = (ytest.transpose() * log(1 + exp(-tm)) +
                      (1 - ytest).transpose() * log(1 + exp(tm))) * invntest

            tt = np.mat(np.zeros(mtest)).transpose()
            tt[np.where(tm > 0)[0], 0] = 1
            tstacc = len(np.where(tt == ytest)[0]) * invntest

            rcd[i] = i, trnerr, tsterr, trnacc, tstacc
        return ww, rcd

    def ploterr(self, dat):
        self.set_111plt((10, 7))
        self.ax.plot(dat[:, 0], dat[:, 1], label="Train",
                     **next(self.keysiter))
        self.ax.plot(dat[:, 0], dat[:, 2], label="Test", **next(self.keysiter))
        self.add_x_labels(cycle(["Iterations"]), self.ax)
        self.add_y_labels(cycle(["Loss"]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig("figQ6a.png", **self.figsave)

    def plotacc(self, dat):
        self.set_111plt((10, 7))
        self.ax.plot(dat[:, 0], dat[:, 3], label="Train",
                     **next(self.keysiter))
        self.ax.plot(dat[:, 0], dat[:, 4], label="Test",
                     **next(self.keysiter))
        self.add_x_labels(cycle(["Iterations"]), self.ax)
        self.add_y_labels(cycle(["Accuracy"]), self.ax)
        self.set_tick_size(self.ax)
        self.add_legends(self.ax)
        self.fig.savefig("figQ6b.png", **self.figsave)


if __name__ == '__main__':
    drv = pb6()
    drv.qa()
