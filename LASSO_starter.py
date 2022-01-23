#!/usr/bin/env python
# coding: utf-8

# In[1]:


def ridge_regress( A, y, la_array ):
    sq_mtrx = At@At.T # want to invert 235 x 235
    identity = np.identity(len(sq_mtrx))
    n = A.shape[1]
    w = np.zeros((n,1))
    num_lam = len(la_array)
    X = np.zeros((n, num_lam))
    
    # Compute solution set for a range of lambda values
    for i, each_lambda in enumerate(la_array):
        w = A.T@np.linalg.inv(sq_mtrx + each_lambda*identity)@y
        X[:, i:i+1] = w

    return X


# In[2]:


def ista_solve_hot( A, d, la_array ):
    # ista_solve_hot: Iterative soft-thresholding for multiple values of
    # lambda with hot start for each case - the converged value for the previous
    # value of lambda is used as an initial condition for the current lambda.
    # this function solves the minimization problem
    # Minimize |Ax-d|_2^2 + lambda*|x|_1 (LASSO regression)
    # using iterative soft-thresholding.
    
    max_iter = 10**4
    tol = 10**(-3)
    tau = 1/np.linalg.norm(A,2)**2
    n = A.shape[1]
    w = np.zeros((n,1))
    num_lam = len(la_array)
    X = np.zeros((n, num_lam))
    for i, each_lambda in enumerate(la_array):
        for j in range(max_iter):
            z = w - tau*(A.T@(A@w-d))
            w_old = w
            w = np.sign(z) * np.clip(np.abs(z)-tau*each_lambda/2, 0, np.inf)
            X[:, i:i+1] = w
            if np.linalg.norm(w - w_old) < tol:
                break
    return X


# In[3]:


def pred_error( A, y, w_param ):
    err_rate = []
    
    for i in range(len(w_param[0])):
        y_pred = A@w_param[:,[i]]
        error_vec = [0 if k[0]==k[1] else 1 for k in np.hstack((np.sign(y_pred), y))]
        err_rate.append(sum(error_vec)/len(y[0]))
    
    return err_rate


# In[9]:


## Breast Cancer LASSO Exploration
## Prepare workspace

from scipy.io import loadmat
import numpy as np
X = loadmat("BreastCancer.mat")['X']
y = loadmat("BreastCancer.mat")['y']

##  10-fold CV 

# each row of setindices denotes the starting an ending index for one
# partition of the data: 5 sets of 30 samples and 5 sets of 29 samples
setindices = [[1,30],[31,60],[61,90],[91,120],[121,150],[151,179],[180,208],[209,237],[238,266],[267,295]]

# each row of holdout indices denotes the partitions that are held out from
# the training set
holdoutindices = [[1,2],[2,3],[3,4],[4,5],[5,6],[7,8],[9,10],[10,1]]

cases = len(holdoutindices)

lams = [1e-6,1e-4,1e-2,1e-1]
lams = np.hstack((lams, np.logspace(0,2,num=20)))
sum_sqe_lasso = 0
sum_sqe_ridge = 0
sum_err_lasso = 0
sum_err_ridge = 0

# Loop over various cases
for j in range(cases):
    # row indices of first validation set
    v1_ind = np.arange(setindices[holdoutindices[j][0]-1][0]-1,setindices[holdoutindices[j][0]-1][1])
    
    # row indices of second validation set
    v2_ind = np.arange(setindices[holdoutindices[j][1]-1][0]-1,setindices[holdoutindices[j][1]-1][1])
    
    # row indices of training set
    trn_ind = list(set(range(295))-set(v1_ind)-set(v2_ind))
    
    # define matrix of features and labels corresponding to first
    # validation set
    Av1 = X[v1_ind,:]
    bv1 = y[v1_ind]
    
    # define matrix of features and labels corresponding to second
    # validation set
    Av2 = X[v2_ind,:]
    bv2 = y[v2_ind]
    
    # define matrix of features and labels corresponding to the 
    # training set
    At = X[trn_ind,:]
    bt = y[trn_ind]
    

# Use training data to learn classifier
    w_lasso_param = ista_solve_hot(At,bt,lams)
    w_ridge_param = ridge_regress(At,bt,lams)
    
    # Compute prediction error for LASSO and ridge regression weights
    err_rate_lasso = pred_error(Av1,bv1,w_lasso_param)
    min_err =  min(err_rate_lasso) # Locate the minimum
    w_lasso_opt = w_lasso_param[:,(err_rate_lasso.index(min_err))].reshape(len(X[0]),1)
    
    err_rate_ridge = pred_error(Av1,bv1,w_ridge_param)
    min_err =  min(err_rate_ridge) # Locate the minimum
    w_ridge_opt = w_ridge_param[:,(err_rate_ridge.index(min_err))].reshape(len(X[0]),1)
    
    # Perform on second hold out set for optimum LASSO and ridge regressions weights
    y_pred = Av2@w_lasso_opt
    error_vec_lasso = [0 if k[0]==k[1] else 1 for k in np.hstack((np.sign(y_pred), bv2))]
    
    y_pred = Av2@w_ridge_opt
    error_vec_ridge = [0 if k[0]==k[1] else 1 for k in np.hstack((np.sign(y_pred), bv2))]
    
    # Calculate error rate
    last_err_lasso = sum(error_vec_lasso)/len(bv2)
    sum_err_lasso += last_err_lasso
    
    last_err_ridge = sum(error_vec_ridge)/len(bv2)
    sum_err_ridge += last_err_ridge
    
    # Calculate squared error
    sq_err_lasso = np.linalg.norm(Av2@w_lasso_opt - bv2)**2
    sum_sqe_lasso += sq_err_lasso
    
    sq_err_ridge = np.linalg.norm(Av2@w_ridge_opt - bv2)**2
    sum_sqe_ridge += sq_err_ridge

# Find best lambda value using first validation set, then evaluate
# performance on second validation set, and accumulate performance metrics
# over all cases partitions

print("Average LASSO squared error:", sum_sqe_lasso / 8)
print("Average LASSO error rate percentage:", (sum_err_lasso / 8) * 100)

print("Average ridge regression squared error:", sum_sqe_ridge / 8)
print("Average ridge regression error rate percentage:", (sum_err_ridge / 8) * 100)

