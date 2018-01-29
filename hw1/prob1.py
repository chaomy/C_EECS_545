#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-25 22:15:39
# @Last Modified by:   chaomy
# @Last Modified time: 2018-01-29 00:09:42


# gradient = A^t (A w - b) -> grd = xx.transpose() * (xx * ww - yy)

import numpy as np
import hw1
from sys import stdout
from itertools import cycle


class pb1(hw1.hw1):

    def __init__(self):
        hw1.hw1.__init__(self)
        self.mxstep = 300
        self.normalize_train_and_test()

    def qa(self):
        stdout.write("after normalize : mean {:04f}\n".format(
            np.mean(self.xtrain)))
        stdout.write("after normalize : std {:04f}\n".format(
            np.std(self.xtrain)))

    def bgd(self, phi, yy):
        ww = np.mat(-0.1 + 0.2 * np.random.rand(phi.shape[1], 1))
        eta, mxstep = 5e-4, self.mxstep
        ls = np.mean(np.power((phi * ww - yy), 2)) + 100
        rcd = np.ndarray([mxstep, 2])
        for i in range(mxstep):
            tm = phi * ww - yy
            cr = np.mean(np.power(tm, 2))
            ww = ww - eta * phi.transpose() * tm
            ls = cr
            rcd[i, :] = i, cr
        rcd = rcd[:i, :]
        return ls, ww, rcd

    def sgd(self, phi, yy):
        (m, n) = phi.shape
        ww = np.mat(-0.1 + 0.2 * np.random.rand(n, 1))
        eta, mxstep = 5e-4, self.mxstep
        ls = np.mean(np.power((phi * ww - yy), 2)) + 100
        ix = np.arange(m)
        rcd = np.ndarray([mxstep, 2])
        for i in range(mxstep):
            np.random.shuffle(ix)
            sm = 0.0
            for j in ix:
                tm = phi[j] * ww - yy[j]
                cr = np.power(tm[0, 0], 2)
                sm += cr
                ww = ww - (eta * phi[j].transpose() * tm)
            ls = sm / m
            rcd[i, :] = i, ls
        rcd = rcd[:i, :]
        return ls, ww, rcd

    def closedform(self, phi, yy):  # use svd find Moore-Penrose Pseudoinverse
        # use pinv
        # ww = np.linalg.pinv(phi) * yy

        uu, sv, vv = np.linalg.svd(phi)
        ss = np.zeros(phi.transpose().shape, dtype=complex)
        ss[:, :len(sv)] = np.diag(1. / sv)
        p = vv.transpose() * ss * uu.transpose()
        ww = p * yy   # closed form solution

        ls = np.real(np.mean(np.power((phi * ww - yy), 2)))
        return ls, ww

    def myout(self, e1, e2, ww, lb):
        stdout.write(
            "Using {}, train error: {:.5f} test error: {:.5f} train/test {:.5f}\n".format(lb, e1, e2, e1 / e2))
        stdout.write("weights: ")
        np.savetxt(stdout, np.real(ww[:-1]).transpose(), fmt="%.4f")
        stdout.write("bias: ")
        np.savetxt(stdout, np.real(ww[-1]).transpose(), fmt="%.4f")

    def qb(self):  # sgd
        xx, yy = self.get_train_data()
        phi = self.get_feature(xx, 1)
        (mse, ww, rcd) = self.sgd(phi, yy)

        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, 1)
        er = np.mean(np.power((phi * ww - yy), 2))
        self.rcdsgd = rcd
        self.myout(mse, er, ww, "SGD")

    def qc(self):  # bgd
        xx, yy = self.get_train_data()
        phi = self.get_feature(xx, 1)

        (mse, ww, rcd) = self.bgd(phi, yy)

        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, 1)
        er = np.mean(np.power((phi * ww - yy), 2))
        self.rcdbgd = rcd
        self.ploterr(50)
        self.ploterr(300)
        self.myout(mse, er, ww, "BGD")

    def ploterr(self, stp=300):
        self.set_111plt()
        for dat, lb in zip([self.rcdsgd, self.rcdbgd],
                           ["SGD", "BGD"]):
            self.ax.plot(dat[:, 0][:stp], dat[:, 1][:stp],
                         label=lb, **next(self.keysiter))
            self.add_x_labels(cycle(["epochs"]), self.ax)
            self.add_y_labels(cycle(["mse"]), self.ax)
            self.set_tick_size(self.ax)
            self.add_legends(self.ax)
        self.fig.savefig("figQ1bc{:03d}.png".format(stp), **self.figsave)

    def qd(self):
        xx, yy = self.get_train_data()
        phi = self.get_feature(xx, 1)

        (mse, ww) = self.closedform(phi, yy)

        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, 1)
        er = np.real(np.mean(np.power((phi * ww - yy), 2)))
        self.myout(mse, er, ww, "Closed Form")

    def qe(self):
        m = self.features.shape[0]
        mde, mte = 0.0, 0.0
        npt = 10
        for k in range(npt):
            rmd = np.random.permutation(m)
            features = self.features[rmd]
            labels = self.labels[rmd]

            nsplit = 50
            xtrain, ytrain = features[:-nsplit], labels[:-nsplit]
            xtest, ytest = features[-nsplit:], labels[-nsplit:]

            xtrain, mn, st = self.normalize(xtrain)
            xtest, mn, st = self.normalize(xtest, mn, st)

            phi = self.get_feature(xtrain, 1)
            mse, ww = self.closedform(phi, ytrain)

            phi = self.get_feature(xtest, 1)
            er = np.real(np.mean(np.power((phi * ww - ytest), 2)))

            stdout.write(
                "{} train error: {:.5f} test error: {:.5f} train/test {:.5f}\n".format(k, mse, er, mse / er))
            mde += mse
            mte += er
        stdout.write("Mean training error: {:.5f}, Mean test error: {:.5f} train/test: {:.5f}\n".format(
            mde / npt, mte / npt, mde / mte))
        return


if __name__ == '__main__':
    drv = pb1()
    drv.qa()
    drv.qb()
    drv.qc()
    drv.qd()
    drv.qe()
