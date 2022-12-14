#!/usr/bin/env python
import numpy as np
from numpy.linalg import pinv

"""
Assignment: Linear and Logistic Regression
Section: Linear Regression with radial basis functions

This file contains the main work to be done.
The functions are:
- TODO get_centers_and_sigma: Compute the centers as explained in the assignement hand-out sheet
- TODO design_matrix: Create the design matrix including the rbf expansions and the constant feature
- TODO train: find the analytical solution of linear regression
- TODO compute_error: return the cost function of linear regression Mean Squared Error
- train_and_test: call the compute error function and all sets and return the corresponding errors

"""


def get_centers_and_sigma(n_centers):
    """
    Creates for a given center numbers the numpy array containing the centers and provides a good sigma
    :param n_centers:
    :return:
    """

    ######################
    #
    # TODO
    #
    # TIPs:
    #   - Use the linspace function from numpy
    #

    centers = np.linspace(-1,1,num=n_centers) # TODO: Change me
    sigma = 2/(n_centers+1)  #l is the number of basis function, since the first basis function is equl to 1, there n_centers+1 basis functions
    

    # END TODO
    ######################

    return centers, sigma


def design_matrix(x, centers, sigma):
    """
    Creates the design matrix given the data x.
    The design matrix is built out of radial basis functions.
    Those are family of Gaussians of width sigma, each of them in centered at one of the centers specified in 'centers'.
    The first column is one for all input data.

    E.g: for the data x = [0,1,2], the centers [0,1] and sigma = 1/sqrt(2)
    the function should return: [[1, exp(0), exp(-1)],
								 [1, exp(-1), exp(0)],
								 [1, exp(-4), exp(-1)]] 

    :param x: numpy array of shape (N,1)
    :param centers: List of centers
    :param sigma: parameter to control the width of the RBF
    :return: Expanded data in a numpy array of shape (N,n_centers+1)
    """

    ######################
    #
    # TODO
    #
    # Return the numpy array of shape (N,n_centers+1)
    # Storing the data of the form exp(- (x_i - c_j) ^2 / (2 sigma^2) ) at row i and column j+1
    # Look at the function description for more info
    #
    # TIP: don't forget that the first column has only ones
    #
 
    #convert to flattened vector, needed for writing the values easily, X[:,i]
    x = x.flatten()
 
    
    N = x.size
    n_centers = centers.size
    
    X = np.ones((N,n_centers+1)) #using ones, the first col is already filled with 1

    for i in range(1,n_centers+1): #start at 1 (not 0) because the first col is already correct
        X[:,i] = np.exp(-((x - centers[i-1])**2)/(2*sigma**2)) #formual of RBF (Radial basis function)
    
   
    
    res = X

    # END TODO
    ######################

    return res


def train(x, y, n_centers):
    """
    Returns the optimal coefficients theta that minimizes the error
    ||  X * theta - y ||**2
    when X is the RBF expansion of x_train with n_centers being the number of kernel centers.

    :param x: input data as numpy array
    :param y: output data as numpy array
    :param n_centers: number of cluster centers
    :return: numpy array containing the coefficients of each polynomial degree in the regression
    """

    ######################
    #
    # TODO
    #
    # Return the analytical solution of the linear regression
    #
    # TIPs:
    #   - Don't forget to first expand the data
    #   - This should not be very different from the solution you provided in poly.py
    #
    

    centers, sigma = get_centers_and_sigma(n_centers)
    
    X = design_matrix(x, centers, sigma)
    X_pinv = np.linalg.pinv(X)
    
    theta_opt = np.zeros(n_centers + 1)
    theta_opt = np.dot(X_pinv,y)


      

    # END TODO
    ######################

    return theta_opt


def compute_error(theta, n_centers, x, y):
    """
    Predicts the value of y given by the model given by theta and number of centers.
    Then compares the predicted value to y and provides the Mean squared error.

    :param theta: Coefficients of the linear regression
    :param n_centers: Number of RBF centers in the RBF expansion
    :param x: Input data
    :param y: Output data to be compared to prediction
    :return: err: Mean squared error
    """

    ######################
    #
    # TODO
    #
    # Return the error (i.e. the cost function)
    #
    # TIPs:
    #   - Don't forget to first expand the data
    #   - This should not be very different from the solution you provided in poly.py
    #
    
    centers, sigma = get_centers_and_sigma(n_centers)
    
    X = design_matrix(x, centers, sigma)
        
    #residual error r
    r = y - np.dot(X,theta)     
    
    #cost function = error
    MSE = np.mean(r**2)
    

    err = MSE  

    # END TODO
    ######################

    return err


def train_and_test(data, n_centers):
    """
    Trains the model with the number of centers 'n_centers' and provides the MSE for the training, validation and testing
     sets

    :param data:
    :param n_centers: number of centers
    :return:
    """

    theta = train(data['x_train'], data['y_train'], n_centers)

    err_train = compute_error(theta, n_centers, data['x_train'], data['y_train'])
    err_val = compute_error(theta, n_centers, data['x_val'], data['y_val'])
    err_test = compute_error(theta, n_centers, data['x_test'], data['y_test'])

    return theta, err_train, err_val, err_test
    
    
if __name__ == "__main__": #if the file was called directly via cmd, execute this partition
    pass
    
    # x = np.array([0,1,2])
    # n_centers = 3
    
    # y = np.array([1, 10, -2])
    
    # theta_opt = train(x, y, n_centers)
    # print(theta_opt)
    
    # print(compute_error(theta_opt, n_centers, x, y))
    
