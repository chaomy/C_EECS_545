#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-10 15:31:54
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-12 02:11:11

from matplotlib.ticker import NullFormatter
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import hw2


class bayesian_linear_regression(hw2.hw2):
    # we defined a class for sequential bayesian learner

    # initialized with covariance matrix(sigma), mean vector(mu) and
    # prior(beta)
    def __init__(self, sigma, mu, beta):
        hw2.hw2.__init__(self)
        self.sigma = sigma
        self.mu = mu
        self.beta = beta
        self.tw = (0.5, -0.3)

    # you need to implement the update function
    # when received additional design matrix phi and continuous label t
    def update(self, phi, t):
        # S_i = ( S_i-1^-1 + beta * phi^T phi)^(-1)
        # m_i = S_i * (beta * phi^T t + S_i-1^-1 m_i-1)
        beta = self.beta
        invlastS = np.linalg.inv(self.sigma)
        self.sigma = np.linalg.inv(
            invlastS + beta * phi.transpose() * phi)
        self.mu = self.sigma * (beta * phi.transpose()
                                * t + invlastS * self.mu)
        return self.mu, self.sigma

    def plt2Dmargin(self, mu, sigma, fnm='fig_2D.png'):
        self.set_111plt((8, 8))
        plot_delta = 0.025
        xx = np.arange(-3.0, 3.0, plot_delta)
        yy = np.arange(-3.0, 3.0, plot_delta)
        X, Y = np.meshgrid(xx, yy)
        Z = np.zeros((xx.shape[0], yy.shape[0]))

        n = xx.shape[0]
        invsig = np.linalg.inv(sigma)
        for i in range(n):
            for j in range(n):
                ps = np.mat([X[i, j], Y[i, j]]).transpose()
                Z[i, j] = np.exp(-0.5 * (ps - mu).transpose() *
                                 invsig * (ps - mu))
        Z = Z / np.sqrt(np.power(2. * np.pi,
                                 mu.shape[0]) * np.linalg.det(sigma))
        nullfmt = NullFormatter()         # no labels

        # definitions for the axes
        thick = 0.10
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02

        rec2d = [left, bottom, width, height]
        rec1dx = [left, bottom_h, width, thick]
        rec1dy = [left_h, bottom, thick, height]

        plt.figure(figsize=(8, 8))

        ax2d = plt.axes(rec2d)
        ax1x = plt.axes(rec1dx)
        ax1y = plt.axes(rec1dy)

        ax1x.xaxis.set_major_formatter(nullfmt)
        ax1y.yaxis.set_major_formatter(nullfmt)

        extent = [-3.0, 3.0, -3.0, 3.0]

        cs = ax2d.contour(X, Y, Z, extent=extent, origin='upper')
        self.ax.clabel(cs, inline=0.1, lw=5, fontsize=16,
                       extent=extent, colors=self.colorlist)
        cs = ax2d.imshow(Z, interpolation='none',
                         extent=extent, origin='lower', cmap="plasma")
        ax1x.plot(xx, mlab.normpdf(xx, mu[0, 0], sigma[0, 0]),
                  lw=4, color=self.colorlist[0])
        ax1y.plot(mlab.normpdf(yy, mu[1, 0], sigma[1, 1]), yy,
                  lw=4, color=self.colorlist[1])
        ax2d.plot(self.tw[0], self.tw[1], '*', color='c', markersize=10)
        plt.savefig(fnm)


def data_generator(size, scale):
    x = np.random.uniform(low=-3, high=3, size=size)
    rand = np.random.normal(0, scale=scale, size=size)
    y = 0.5 * x - 0.3 + rand
    phi = np.array([[x[i], 1] for i in range(x.shape[0])])
    t = y
    return phi, t


def main():
    # initialization
    alpha = 2
    sigma_0 = np.mat(np.diag(1.0 / alpha * np.ones([2])))
    mu_0 = np.mat(np.zeros([2])).transpose()
    beta = 1.0
    blr_learner = bayesian_linear_regression(sigma_0, mu_0, beta=beta)

    num_episodes = 20
    blr_learner.plt2Dmargin(
        mu_0, sigma_0, "figQ3{:03}.png".format(0))

    for epi in range(num_episodes):
        phi, t = data_generator(1, 1.0 / beta)
        mu, sigma = blr_learner.update(phi, t)
        if epi in [0, 9, 19]:
            blr_learner.plt2Dmargin(
                mu, sigma, "figQ3{:03}.png".format(epi + 1))


if __name__ == '__main__':
    main()
