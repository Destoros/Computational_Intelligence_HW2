#!/usr/bin/env python
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

import gradient_descent as gd
import logreg as lr
import logreg_toolbox

"""
Assignment: Linear and Logistic Regression
Section: Gradient descent (GD) and Logistic Regression

This is the main file that loads the data, computes the solution and plots the results.
"""


def main():
    # Set parameters
    degree = 1
    eta = 2
    max_iter = 200

    testset = False

    if testset:
        setgroup = "test"
    else:
        setgroup = "train"


    # Load data and expand with polynomial features
    f = open('data_logreg.json', 'r')
    data = json.load(f)
    for k, v in data.items(): data[k] = np.array(v)  # Encode list into numpy array

    # Expand with polynomial features
    X_train = logreg_toolbox.poly_2D_design_matrix(data['x1_'+setgroup], data['x2_'+setgroup], degree)
    n = X_train.shape[1]

    # Define the functions of the parameter we want to optimize
    def f(theta): return lr.cost(theta, X_train, data['y_'+ setgroup])

    def df(theta): return lr.grad(theta, X_train, data['y_'+setgroup])

    # Test to verify if the computation of the gradient is correct
    logreg_toolbox.check_gradient(f, df, n)

    # Point for initialization of gradient descent

    theta0 = np.zeros(n)
    theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta, max_iter)

    # eta_step = 1
    # iter_step = 300
    #
    # findIter = True
    # old_iter_error = 1
    # counter = 0
    # while findIter == True:
    #     eta = 0.1
    #     findEta = True
    #     old_eta_error = 1
    #
    #     print(max_iter)
    #     while findEta == True:
    #
    #         theta0 = np.zeros(n)
    #         theta_opt, E_list = gd.gradient_descent(f, df, theta0, eta, max_iter)
    #
    #         print(eta)
    #
    #         if f(theta_opt) > old_eta_error:
    #             findEta = False
    #         else:
    #             old_eta_error = f(theta_opt)
    #             eta = np.round(eta + eta_step, 2)
    #
    #     if f(theta_opt) > old_iter_error:
    #         findIter = False
    #     else:
    #         old_iter_error = f(theta_opt)
    #         max_iter = max_iter + iter_step
    #
    #
    #
    # print("Degree: ",  degree)
    # print("Eta:" , eta)
    # print("Interation" ,max_iter)
    print("Error" , f(theta_opt))

    logreg_toolbox.plot_logreg(data, degree, theta_opt, E_list)
    plt.show()


if __name__ == '__main__':
    main()
