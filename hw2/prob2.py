#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: chaomy
# @Date:   2018-02-09 23:47:35
# @Last Modified by:   chaomy
# @Last Modified time: 2018-02-12 14:50:40

import hw2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import matplotlib.cm as cm
from sys import stdout
from itertools import cycle
from matplotlib.ticker import NullFormatter


# feel free to read the two examples below, try to understand them
# in this problem, we require you to generate contour plots

# generate contour plot for function z = x^2 + 2*y^2
def plot_contour():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    plt.axis("square")
    cs = plt.contour(X, Y, Z)
    plt.clabel(cs, inline=0.1, fontsize=10)
    plt.show()


# generate heat plot (image-like) for function z = x^2 + 2*y^2
def plot_heat():
    plot_delta = 0.025
    plot_x = np.arange(-3.0, 3.0, plot_delta)
    plot_y = np.arange(-3.0, 3.0, plot_delta)
    X, Y = np.meshgrid(plot_x, plot_y)
    x_len = plot_x.shape[0]
    y_len = plot_y.shape[0]
    Z = np.zeros((x_len, y_len))
    for i in range(x_len):
        for j in range(y_len):
            Z[j][i] = X[j][i] ** 2 + 2 * (Y[j][i] ** 2)

    plt.clf()
    img = Z
    plt.imshow(img, interpolation='none',
               extent=[-3.0, 3.0, -3.0, 3.0], cmap="plasma")
    plt.colorbar()
    plt.axis("square")
    plt.show()

test_sigma_1 = np.mat(np.array(
    [[1.0, 0.5],
     [0.5, 1.0]]
))

test_mu_1 = np.mat(np.array(
    [0.0, 0.0]
))

test_sigma_2 = np.mat(np.array(
    [[1.0, 0.5, 0.0, 0.0],
     [0.5, 1.0, 0.0, 1.5],
     [0.0, 0.0, 2.0, 0.0],
     [0.0, 1.5, 0.0, 4.0]]
))

test_mu_2 = np.mat(np.array(
    [0.5, 0.0, -0.5, 0.0]
))

indices_1 = np.array([0])
indices_2 = np.array([1, 2])
values_2 = np.array([0.1, -0.2])


class pb2(hw2.hw2):

    def marginal_for_guassian(self, sigma, mu, given_indices):
        # This function receives the parameters of a multivariate Gaussian distribution
        # over variables x_1, x_2 .... x_n as input and compute the marginal
        # given selected indices, compute marginal distribution for them
        # N(xa | mu_a, sigma_aa)
        sigma_aa = sigma[np.ix_(given_indices, given_indices)]
        mu_a = mu[np.ix_([0], given_indices)].transpose()
        return mu_a, sigma_aa

    def conditional_for_gaussian(self, sigma, mu, given_indices, given_values):
        # given some indices that have fixed value, compute the conditional distribution
        # for rest indices
        # sigma_b|a = sigma_bb - sigma_ba sigma_aa^-1 sigma_ab
        # mu_b|a = mu_b + sigma_ba sigma_aa^-1 (xa - mu_a)
        indxall = np.arange(sigma.shape[0])
        return_indices = np.delete(indxall, given_indices)

        sigma_aa = sigma[np.ix_(given_indices, given_indices)]
        sigma_bb = sigma[np.ix_(return_indices, return_indices)]
        sigma_ab = sigma[np.ix_(given_indices, return_indices)]
        sigma_ba = sigma[np.ix_(return_indices, given_indices)]

        mu_a = mu[np.ix_([0], given_indices)].transpose()
        mu_b = mu[np.ix_([0], return_indices)].transpose()
        x_a = np.mat(given_values).transpose()

        sigma_b_a = sigma_bb - sigma_ba * np.linalg.inv(sigma_aa) * sigma_ab
        mu_b_a = mu_b + sigma_ba * np.linalg.inv(sigma_aa) * (x_a - mu_a)
        return (mu_b_a, sigma_b_a)

    def qb(self):
        (mu_a, sigma_aa) = self.marginal_for_guassian(
            test_sigma_1, test_mu_1, indices_1)
        self.pltmargin(test_mu_1.transpose(), test_sigma_1, fnm="figQ2b.png")

    def qc(self):
        self.conditional_for_gaussian(
            test_sigma_2, test_mu_2, indices_2, values_2)

    def qd(self):
        (mu_b_a, sigma_b_a) = self.conditional_for_gaussian(
            test_sigma_2, test_mu_2, indices_2, values_2)

        stdout.write("conditional mean vectors\n")
        np.savetxt(stdout, mu_b_a, fmt="%.3f")
        stdout.write("conditional covariance matrices\n")
        np.savetxt(stdout, sigma_b_a, fmt="%.3f")

        self.plt2Dcondition(mu_b_a, sigma_b_a, fnm="figQ2d.png", lbs=["X1", "X4"])

    def pltmargin(self, mu, sigma, fnm='fig_2D.png', lbs=["X1", "P(X1)"]):
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
        plt.figure(figsize=(8, 8))
        ax2d = plt.axes()
        ax2d.set_xlabel(lbs[0], {'fontsize': self.myfontsize})
        ax2d.set_ylabel(lbs[1], {'fontsize': self.myfontsize})
        ax2d.plot(xx, mlab.normpdf(xx, mu[0, 0], sigma[0, 0]),
                  lw=4, color=self.colorlist[0])
        plt.savefig(fnm)

    def plt2Dcondition(self, mu, sigma, fnm='fig_2D.png', lbs=["X1", "X2"]):
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
        plt.figure(figsize=(8, 8))
        ax2d = plt.axes()
        ax2d.set_xlabel(lbs[0], {'fontsize': self.myfontsize})
        ax2d.set_ylabel(lbs[1], {'fontsize': self.myfontsize})
        extent = [-3.0, 3.0, -3.0, 3.0]
        cs = ax2d.contour(X, Y, Z, extent=extent, origin='upper')
        self.ax.clabel(cs, inline=0.1, lw=5, fontsize=16,
                       extent=extent, colors=self.colorlist)
        cs = ax2d.imshow(Z, interpolation='none',
                         extent=extent, origin='lower', cmap="plasma")
        plt.savefig(fnm)

    def plt2Dmargin(self, mu, sigma, fnm='fig_2D.png', lbs=["X1", "X2"]):
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
        thick = 0.1
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

        ax2d.set_xlabel(lbs[0], {'fontsize': self.myfontsize})
        ax2d.set_ylabel(lbs[1], {'fontsize': self.myfontsize})

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
        plt.savefig(fnm)


if __name__ == '__main__':
    drv = pb2()
    drv.qb()
    drv.qc()
    drv.qd()
