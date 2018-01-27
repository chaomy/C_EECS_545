#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-25 22:15:39
# @Last Modified by:   chaomingyang
# @Last Modified time: 2018-01-27 17:52:41


# w0 + w1 * x1 + w2 * x2 ... + w13 * x13
# err = xx * ww - yy
# gradient = A^t (A w - b) -> grd = xx.transpose() * (xx * ww - yy)


import numpy as np
import hw1


class pb3(hw1.hw1):

    def __init__(self):
        hw1.hw1.__init__(self)

    def bgd(self, xx, yy, lmd=0.0):
        ww = np.mat(-0.1 + 0.2 * np.random.rand(xx.shape[1], 1))
        eta, mxstep = 5e-4, 500
        ls = np.mean(np.power((xx * ww - yy), 2)) + lmd * np.power(ww, 2) + 100

        rcd = np.ndarray([mxstep, 2])
        for i in range(mxstep):
            tm = xx * ww - yy
            cr = np.mean(np.power(tm, 2)) + lmd * np.power(ww, 2)
            if (ls - cr) < 1e-2:
                ls = cr
                break
            # print i, ls, cr, ls - cr
            ww = ww - eta * (xx.transpose() * tm + lmd * ww)
            ls = cr
            rcd[i, :] = i, cr
        rcd = rcd[:i, :]
        return ls, ww, rcd

    def qb(self):
        xx, yy = self.get_train_data()
        bgd(xx, yy, )
        return


if __name__ == '__main__':
    drv = pb2()
    drv.qa()
    drv.qb()
