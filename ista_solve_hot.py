#!/usr/bin/env python
# coding: utf-8

# In[1]:


## Breast Cancer LASSO Exploration
## Prepare workspace

from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt 

X = loadmat("BreastCancer.mat")['X']
y = loadmat("BreastCancer.mat")['y']

X_100 = X[0:100]
y_100 = y[0:100]

lams = [1e-6,1e-4,1e-2,1e-1]
print(lams)
lams = np.hstack((lams, np.logspace(0,2,num=20)))
print(lams)


# In[2]:


def ista_solve_hot( A, d, la_array ):
    # ista_solve_hot: Iterative soft-thresholding for multiple values of
    # lambda with hot start for each case - the converged value for the previous
    # value of lambda is used as an initial condition for the current lambda.
    # This function solves the minimization problem:
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


# In[10]:


w_param = ista_solve_hot(X_100, y_100, lams)
res = []
l1_reg = []
loss_func = []
min_loss = None
min_idx = None

for i in range(len(w_param[0])):
    res.append(np.linalg.norm(X_100@w_param[:,[i]] - y_100))
    l1_reg.append(np.linalg.norm(w_param[:,[i]], 1))

plt.plot(l1_reg, res, 'b.-')


# The left side of the graph demonstrates that a large value of lambda contributes to a greater residual.
# On the other hand, the right side is when lambda is small, meaning a low residual and large 1 norm. 

# In[4]:


err_rate = []
sparsity = []

for i in range(len(w_param[0])):
    # Calculate error rate
    y_pred = X_100@w_param[:,[i]]
    error_vec = [0 if k[0]==k[1] else 1 for k in np.hstack((np.sign(y_pred), y_100))]
    err_rate.append(sum(error_vec)/len(y_100[0]))
    
    # Calculate sparsity
    nz_entries = len([k for k in w_param[:, i] if k >= 1e-6])
    sparsity.append(nz_entries)

plt.plot(sparsity, err_rate, 'b.-')
plt.xlim([0, 150])


# When the sparsity is low, the error rate sky rockets as opposed to when there are more non-zero weight components which do a 
# better job at classifying data points.

# In[26]:


X_101_end = X[100:295]
y_101_end = y[100:295]

res = []
l1_reg = []
loss_func = []
min_loss = None
min_idx = None

for i in range(len(w_param[0])):
    res.append(np.linalg.norm(X_101_end@w_param[:,[i]] - y_101_end))
    l1_reg.append(np.linalg.norm(w_param[:,[i]], 1))

plt.plot(l1_reg, res, '-b.')


# #### 1ci)
# 
# Using our parameterized weights on validatation data, we notice that the 15th weight vector in w* has the least residual. When lambda is 0, LASSO effectively computes the least squares solution so it may be computatatively beneficial to shrink variable coefficients seeing as many of the lambda values greater than 1e-6 have less residual error.

# In[28]:


err_rate = []
sparsity = []

for i in range(len(w_param[0])):
    y_pred = X_101_end@w_param[:,[i]]

    error_vec = [0 if k[0]==k[1] else 1 for k in np.hstack((np.sign(y_pred), y_101_end))]
    err_rate.append(sum(error_vec)/len(y_100[0]))
    
    nz_entries = len([k for k in w_param[:, i] if k >= 1e-6])
    sparsity.append(nz_entries)

plt.plot(sparsity, err_rate, '-b.')
plt.xlim([0, 150])

# We can see a large lambda doesn't make for the best error rate. Every other lambda has a similar effect on the error.
