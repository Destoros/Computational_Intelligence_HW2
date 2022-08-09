#!/usr/bin/env python
import numpy as np
from numpy.linalg import pinv

"""
Assignment: Linear and Logistic Regression
Section: Linear Regression with polynomial features

This file contains the main work to be done.
The functions are:
- TODO design_matrix: Create the design matrix including the polynomial expansions and the constant feature
- TODO train: finds the analytical solution of linear regression
- TODO compute_error: return the cost function of linear regression Mean Square Error
- train_and_test: call the compute error function and all sets and return the corresponding errors

"""


def design_matrix(x, degree):
    """
    Creates the design matrix given the data x.
    The design matrix is built of all polynomials of x from degree 0 to 'degree'.

    EX: for the data x = [0,1,2] and degree 2
    the function should return: [[1, 0, 0],
								 [1, 1, 1],
								 [1, 2, 4]] 

    :param x: numpy array of shape (N,1)
    :param degree: Higher degree of the polynomial
    :return: Expanded data in a numpy array of shape (N,degree+1)
    """

    ######################
    #
    # TODO
    #
    # Return the numpy array of shape (N,degree+1)
    # Storing the data of the form x_i^j at row i and column j
    # Look at the function description for more info
    #
    # TIP: use the power function from numpy
    
    #convert to flattened vector
    x = x.flatten()
    N = x.size #how many elements are there in x; this allows for filling the the matrix by X[:,i] 
        
    
    X = np.zeros( (N,degree+1) )
    
    for i in range(degree+1): #range(2) returns the array [0, 1]; 2 is not included
        X[:,i] = np.power(x,i)
     
   
 
    

    #
    # END TODO
    ######################

    return X


def train(x, y, degree):
    """
    Returns the optimal coefficients theta that minimizes the error
    ||  X * theta - y ||**2
    when X is the polynomial expansion of x_train of degree 'degree'.

    :param x: numpy array on the input
    :param y: numpy array containing the output
    :param degree: maximum polynomial degree in the polynomial expansion
    :return: a numpy array containing the coefficient of each polynomial degree in the regression
    """

    ######################
    #
    # TODO
    #
    # Returns the analytical solution of the linear regression
    #
    # TIPs:
    #  - Don't forget to first expand the data
    #  - WARNING:   With numpy array * is a term-term matrix multiplication; I guess term-term is elementwise multiplication
    #               The function np.dot performs a classic matrix multiplication (recent Python version accept @)
    #
    #  - To compute the pseudo inverse (A*A.T)^-1 * A.T with a more stable algorithm numpy provides the function pinv
    #   pinv is accessible in the sub-library numpy.linalg
    #
    
    
    X = design_matrix(x, degree) #returns phi matrix as declared in VO
    
    theta_opt = np.zeros(degree + 1)  # TODO: Change me
    X_pinv = np.linalg.pinv(X)
    
    
    theta_opt = np.dot(X_pinv,y)

    # END TODO
    ######################

    return theta_opt


def compute_error(theta, degree, x, y):
    """
    Predicts the value of y given by the model given by theta and degree.
    Then compares the predicted value to y and provides the Mean squared error.

    :param theta: Coefficients of the linear regression
    :param degree: Degree in the polynomial expansion
    :param x: Input data
    :param y: Output data to be compared to prediction
    :return: err: Mean squared error
    """

    ######################
    #
    # TODO
    #
    # Returns the error (i.e. the cost function)
    #
    # TIPs:
    #  - WARNING:   With numpy array * is a term-term matrix multiplication
    #               The function np.dot performs a matrix multiplication
    #               A longer alternative is to first change your array to the matrix class using np.matrix,
    #               Then * becomes a matrix multiplication
    #
    #  - One can use the numpy function mean
    
    
    X = design_matrix(x, degree) 
        
    #residual error r
    r = y - np.dot(X,theta)     
    
    #cost function = error
    MSE = np.mean(r**2)
    # np.linalg.norm does not return the same values as this approach, because the mean has a scaling of 1/N before the sum
    #why is there in the TODO: "One can use the numpy function mean" if mean does not represent the the true cost function which is
    #E(w) = 0.5 * ||r||^2 , where ||( )||^2 denotes the norm operator     
    #If I use the mean approach, the values of MSE fit into the error plot which has a ylim([0, 1]). The np.linalg.norm() approach causes the MSE
    #values to be too big and not fit onto the given y limits

    #Answer from me to myself <3
    #Thats because the >mean< squared error is asked for and not the cost function. I was mislead by the sentence "Returns the error (i.e. the cost function)"
    #Since the given skeleten also labels the returned value err as MSE Im quite confident, this approach is correct now
    
    err = MSE 

    #
    # END TODO
    ######################

    return err


def train_and_test(data, degree):
    """
    Trains the model with degree 'degree' and provides the MSE for the training, validation and testing sets

    :param data:
    :param degree:
    :return:
    """

    theta = train(data['x_train'], data['y_train'], degree)

    err_train = compute_error(theta, degree, data['x_train'], data['y_train'])
    err_val = compute_error(theta, degree, data['x_val'], data['y_val'])
    err_test = compute_error(theta, degree, data['x_test'], data['y_test'])

    return theta, err_train, err_val, err_test


def main():
    # degree = 3
    # x = np.array([[0,1], [2,3]]);
    # print(design_matrix(x, degree))
    
    #test design matrix
    # x = np.array([0, 1, 2])
    # degree = 3
    # print(design_matrix(x, degree))
    
    # ------------------------------------------------------------------------------------
    #test train
    #by creating an polynomial in the weights can be easily checked
    #the polynomial is as follows:
    # y = 1 + 3*x^2 +0.5*x^4
    x = np. array([-5,    -4,    -3,    -2,    -1,     0,     1,     2,     3,     4,     5])
    y = np. array([388.5000,  177.0000,   68.5000,   21.0000,    4.5000,   1.0000,    4.5000,   21.0000,   68.5000,  177.0000,  388.5000])
    # degree = 4
    
    a = 1;

    
    #test, if we know the degree of the polynomial exactly
    # theta_opt = train(x, y, degree)
    # print(theta_opt) 
    
    #theta_opt =[ 1.00000000e+00 -1.59872116e-14  3.00000000e+00  8.88178420e-16  5.00000000e-01]
                     #the values match with the given polynomial, there are some minor errors at the x^1 and x^3 which should be 0, but they are really close to zero
                     
    #now what happens if we over- or undermodel the polynomial, which means the given variable 'degree' does not fit to the actual degree of the polynomial
    
    #overmodelling:
    # degree = 10
    # print(train(x, y, degree))
    
    #theta_opt = [ 1.00000000e+00  3.03016723e-10  3.00000000e+00 -1.58177915e-10  5.00000000e-01  2.41624498e-11 -1.83520421e-11 -1.37666961e-12  1.05093365e-12  2.56836653e-14 -1.97664151e-14]
    #the values of the non zero constants are still really good, but the values of x^1 and x^3 moved further away from zero, which is bad. The tailing values after x^4 are also non zero and could cause
    #   a bad fit especially the further we move away from the origign (because of the higher order terms)
    
    #Now for a extreme overmodelling    
    # degree = 100
    # print(train(x, y, degree))
    # y = 1 + 3*x^2 +0.5*x^4
    # theta_opt = [ 9.99999976e-01 -7.31734959e-04  7.55646152e-02 -7.31734823e-04
                  # 7.55645375e-02 -7.31731237e-04  7.55633247e-02 -7.31640581e-04
                  # 7.55447486e-02 -7.29374184e-04  7.52690994e-02 -6.72856160e-04
                  # 7.14068668e-02  7.36559298e-04 -4.77596656e-03 -4.32166337e-05
                  # 6.28886893e-02 -2.68531768e-03  8.64248150e-02 -2.03009163e-03
                  # 7.63414659e-02 -3.76334119e-04  8.15799377e-02 -2.76251522e-03
                  # 5.41895881e-02 -1.29015725e-03  6.84389343e-02 -2.16493120e-03
                  # 9.43459832e-02  6.69134983e-04  4.30881383e-02  3.61086092e-02
                  # 5.45468083e-02 -1.76713594e-03  7.38226268e-02  1.07816045e-03
                  # 5.61678732e-02 -8.52311749e-04  6.69036264e-02 -2.35454155e-03
                  # 7.66860920e-02 -2.39002650e-04  9.79399375e-02 -2.42579036e-03
                  # 8.35012779e-02  5.52663331e-04  9.24041279e-02 -2.97289713e-03
                  # 6.55573939e-02  1.80198051e-03  7.20876161e-02 -4.47751279e-03
                  # 8.43369943e-02 -2.18364970e-03  6.45966237e-02 -9.70195713e-04
                  # 7.52067880e-02 -5.48668454e-04  4.33466077e-02 -2.11290320e-03
                  # 5.57210545e-02  1.95527492e-03  8.25031054e-02 -2.88455846e-04
                  # 9.91716513e-02 -3.23968008e-03  8.44689403e-02  2.16341016e-03
                  # 1.13192329e-01 -3.66420940e-03  7.44775488e-02 -1.22298157e-04
                  # 7.30335861e-02 -1.65823422e-03  8.17810698e-02  8.14222883e-05
                  # 6.33475069e-02 -1.32344844e-03  7.36405503e-02  2.75350305e-03
                  # 5.47394671e-02 -1.45856134e-03  3.89666562e-02  1.84222563e-03
                  # 4.88138498e-02 -4.93889713e-05  6.04265021e-02  4.57869832e-05
                  # 9.92503005e-02 -2.10736816e-03  7.21054381e-02 -3.92425127e-04
                  # 6.29264322e-02 -8.46148047e-04  7.88665087e-02  3.29132054e-04
                  # 6.88145181e-02  1.73036428e-03  4.49126815e-02 -1.11735590e-04
                  # 5.03011285e-02]
                  
    # The values dont fit at all anymore and wouldnt resemble the true polynomial
    
    
    #Now for undermodelling
    # y = 1 + 3*x^2 +0.5*x^4
    degree = 3 #the degree of the true polynomial is 4
    print(train(x, y, degree))
    
    # theta_opt = [-3.50000000e+01  1.24344979e-14  1.55000000e+01  0.00000000e+00]
    #undermodelling by one already changes the coefficients a lot
    
    #Now for bigger undermodelling 
    # degree = 2 #the degree of the true polynomial is 4
    # print(train(x, y, degree))
    
    #theta_opt = [-3.50000000e+01  1.42108547e-14  1.55000000e+01]
    #it didnt change a lot from the coefficients before
    
    #Now for an even bigger undermodelling 
    # degree = 1 #the degree of the true polynomial is 4
    # print(train(x, y, degree))
    #theta_opt = [1.20000000e+02 7.10542736e-15]
    #worse than before
    
    
    #Conclusio: If the true degree is not know beforehand, it is better to start with overmodelling. A tail of coefficients which are close to zero can be 
    #  a indication for overmodelling. This only holds true for functions which follow a polynomial in the first place.
    
    
    
    
    # ------------------------------------------------------------------------------------
    #test compute error
    #by creating an polynomial in the weights can be easily checked
    #the polynomial is as follows:
    # y = 1 + 3*x^2 +0.5*x^4
    # x = np. array([-5,    -4,    -3,    -2,    -1,     0,     1,     2,     3,     4,     5])
    # y = np. array([388.5000,  177.0000,   68.5000,   21.0000,    4.5000,   1.0000,    4.5000,   21.0000,   68.5000,  177.0000,  388.5000])
    # degree = 4
    
    # theta = train(x, y, degree)
    
    # print(compute_error(theta, degree, x, y))
    pass
    
if __name__ == '__main__':
    main()
    

