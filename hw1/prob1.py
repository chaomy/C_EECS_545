#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-25 22:15:39
# @Last Modified by:   chaomingyang
# @Last Modified time: 2018-01-27 17:50:06


# w0 + w1 * x1 + w2 * x2 ... + w13 * x13
# err = xx * ww - yy
# gradient = A^t (A w - b) -> grd = xx.transpose() * (xx * ww - yy)


import numpy as np
import hw1


class pb1(hw1.hw1):

    def __init__(self):
        hw1.hw1.__init__(self)

    def qa(self):
        return

    def bgd(self, phi, yy):
        ww = np.mat(-0.1 + 0.2 * np.random.rand(phi.shape[1], 1))
        eta, mxstep = 5e-4, 500
        ls = np.mean(np.power((phi * ww - yy), 2)) + 100
        rcd = np.ndarray([mxstep, 2])
        for i in range(mxstep):
            tm = phi * ww - yy
            cr = np.mean(np.power(tm, 2))
            if (ls - cr) < 1e-2:
                ls = cr
                break
            # print i, ls, cr, ls - cr
            ww = ww - eta * phi.transpose() * tm
            ls = cr
            rcd[i, :] = i, cr
        rcd = rcd[:i, :]
        return ls, ww, rcd

    def qb(self):  # sgd
        xx, yy = self.get_train_data()
        phi = self.get_feature(xx, 1)

        (m, n) = phi.shape
        ww = np.mat(-0.1 + 0.2 * np.random.rand(n, 1))

        eta = 5e-4
        ls = np.mean(np.power((phi * ww - yy), 2)) + 100

        ix = np.arange(m)
        # later
        for i in range(100):
            np.random.shuffle(ix)
            sm = 0.0
            for j in ix:
                tm = phi[j] * ww - yy[j]
                cr = np.power(tm[0, 0], 2)
                sm += cr
                ww = ww - (eta * phi[j].transpose() * tm)
                # if (ls - cr) < 1.0:
                #     break
            ls = sm
        return

    def qc(self):  # bgd
        xx, yy = self.get_train_data()
        phi = self.get_feature(xx, 1)

        (mse, ww, rcd) = self.bgd(phi, yy)

        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, 1)
        er = np.mean(np.power((phi * ww - yy), 2))

        self.set_111plt()
        self.ax.plot(rcd[:, 0], rcd[:, 1], **next(self.keysiter))
        self.fig.savefig('fig_q1c.png', **self.figsave)

    #  closed form solutions w = ((A'A)^-1 A') t = pinv(A) * t
    #  use svd find Moore-Penrose Pseudoinverse
    def qd(self):
        xx, yy = self.get_train_data()
        phi = self.get_feature(xx, 1)

        uu, sv, vv = np.linalg.svd(phi)
        ss = np.zeros(phi.transpose().shape, dtype=complex)
        ss[:, :len(sv)] = np.diag(1. / sv)
        p = vv.transpose() * ss * uu.transpose()

        print np.isclose(np.linalg.pinv(phi), p).all()   # check !

        ww = p * yy   # closed form solution
        ls = np.real(np.mean(np.power((phi * ww - yy), 2)))

        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, 1)
        er = np.real(np.mean(np.power((phi * ww - yy), 2)))

        print ls, er

    def qe(self):
        return


if __name__ == '__main__':
    drv = pb1()
    drv.qa()
    # drv.qb()
    drv.qc()
    drv.qd()
