#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-25 22:15:39
# @Last Modified by:   chaomingyang
# @Last Modified time: 2018-01-27 17:52:02

import hw1
import numpy as np


class pb2(hw1.hw1):

    def __init__(self):
        hw1.hw1.__init__(self)

    def train(self, xx, yy, order=2):
        phi = self.get_feature(xx, order)
        ww = np.linalg.pinv(phi) * yy
        rmse = np.sqrt(np.mean(np.power(phi * ww - yy, 2)))
        return rmse, ww

    def test(self, ww, order=2):
        xx, yy = self.get_test_data()
        phi = self.get_feature(xx, order)
        rmse = np.sqrt(np.mean(np.power(phi * ww - yy, 2)))
        return rmse

    def qa(self):
        xx, yy = self.get_train_data()
        for i in range(5):
            dr, ww = self.train(xx, yy, i)
            tr = self.test(ww, i)
            print i, dr, tr
        return

    def qb(self):
        xx, yy = self.get_train_data()
        for rr in [0.2, 0.4, 0.6, 0.8, 1.0]:
            m = int(rr * xx.shape[0])
            dr, ww = self.train(xx[:m], yy[:m], 1)
            tr = self.test(ww, 1)
            print rr, dr, tr
        return

if __name__ == '__main__':
    drv = pb2()
    drv.qa()
    drv.qb()
