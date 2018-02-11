#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-09 14:17:32
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-10 15:32:49

import numpy as np
import pltdrv


class hw2(pltdrv.myplt):

    def __init__(self):
        pltdrv.myplt.__init__(self)

    def normalize(self, xx, mn=None, st=None):
        if mn is None:
            mn = np.mean(xx, axis=0)
            st = np.std(xx, axis=0)

        if xx.shape[1] > 1:
            n0 = np.where(st != 0)
            i0 = np.where(st == 0)
            xx[:, n0] = (xx[:, n0] - mn[:, n0]) / st[:, n0]
            xx[:, i0] = (xx[:, i0] - mn[:, i0])
        else:
            if st == 0:
                xx = (xx - mn) / st
            else:
                xx = xx - mn
        return xx, mn, st

    def normalize_train_and_test(self):
        self.xtrain, mn, st = self.normalize(self.xtrain)
        self.xtest, mn, st = self.normalize(self.xtest, mn, st)

    def get_feature(self, xx, order=1):
        phi = np.ones((xx.shape[0], 1))
        for i in range(1, order + 1):
            phi = np.hstack((np.power(xx, i), phi))
        return phi
