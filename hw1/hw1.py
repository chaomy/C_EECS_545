#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomingyang
# @Date:   2018-01-27 17:30:11
# @Last Modified by:   chaomingyang
# @Last Modified time: 2018-01-27 17:50:53

import numpy as np
import pltdrv
from sklearn import datasets


class hw1(pltdrv.myplt):

    def __init__(self):
        self.dat = {'xtrn': None,
                    'ytrn': None,
                    'xtst': None,
                    'ytst': None}
        self.loaddata()
        self.normalize()
        pltdrv.myplt.__init__(self)

    def loaddata(self):
        dataset = datasets.load_boston()  # Load dataset
        features = dataset.data
        labels = dataset.target
        nsplit = 50
        dat = self.dat
        dat['xtrn'], dat['ytrn'] = np.mat(
            features[:-nsplit]), np.mat(labels[:-nsplit]).transpose()  # Training set
        dat['xtst'], dat['ytst'] = np.mat(
            features[-nsplit:]), np.mat(labels[-nsplit:]).transpose()  # Test set

    def normalize(self):
        print "normalizing data"
        for lb in ['xtrn', 'xtst']:
            xx = self.dat[lb]
            mn = np.mean(xx, axis=0)
            st = np.std(xx, axis=0)

            n0 = np.where(st != 0)[1]
            i0 = np.where(st == 0)[1]

            self.dat[lb][:, n0] = (xx[:, n0] - mn[:, n0]) / st[:, n0]
            self.dat[lb][:, i0] = (xx[:, i0] - mn[:, i0])

    def get_train_data(self):
        return self.dat["xtrn"], self.dat["ytrn"]

    def get_test_data(self):
        return self.dat["xtst"], self.dat["ytst"]

    def get_feature(self, xx, order=1):
        phi = np.ones((xx.shape[0], 1))
        for i in range(1, order + 1):
            phi = np.hstack((np.power(xx, i), phi))
        return phi
