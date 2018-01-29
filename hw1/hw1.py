#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-27 17:30:11
# @Last Modified by:   chaomy
# @Last Modified time: 2018-01-29 00:04:44

import numpy as np
import pltdrv
from sklearn import datasets


class hw1(pltdrv.myplt):

    def __init__(self):
        self.loaddata()
        pltdrv.myplt.__init__(self)

    def loaddata(self):
        dataset = datasets.load_boston()  # Load dataset
        self.features = np.mat(dataset.data)
        self.labels = np.mat(dataset.target).transpose()
        nn = 50
        self.xtrain, self.ytrain = self.features[
            :-nn], self.labels[:-nn]
        self.xtest, self.ytest = self.features[
            -nn:], self.labels[-nn:]

    def normalize_train_and_test(self):
        self.xtrain, mn, st = self.normalize(self.xtrain)
        self.xtest, mn, st = self.normalize(self.xtest, mn, st)

    def normalize(self, xx, mn=None, st=None):
        if mn is None:
            mn = np.mean(xx, axis=0)
            st = np.std(xx, axis=0)

        n0 = np.where(st != 0)[1]
        i0 = np.where(st == 0)[1]

        xx[:, n0] = (xx[:, n0] - mn[:, n0]) / st[:, n0]
        xx[:, i0] = (xx[:, i0] - mn[:, i0])
        return xx, mn, st

    def get_train_data(self):
        return self.xtrain, self.ytrain

    def get_test_data(self):
        return self.xtest, self.ytest

    def get_feature(self, xx, order=1):
        phi = np.ones((xx.shape[0], 1))
        for i in range(1, order + 1):
            phi = np.hstack((np.power(xx, i), phi))
        return phi
