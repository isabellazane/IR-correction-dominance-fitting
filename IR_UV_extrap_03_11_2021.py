# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 22:30:16 2021

@author: Owner
"""

import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import scipy.optimize

f = open('\\Users\\Owner\\Documents\\UV_data_3.txt','r')
line_list =[]
for line in f:
  line = line.strip("\n")
  line_list.append(line)
while '' in line_list:
   line_list.remove('')
line_list = line_list[1:]
line_list = line_list[:11]
nmax = []
nmax_var = []
variance = []
E_res = []
b =[]
for i in line_list:
    nmax.append(i.split('\t')[0])
    nmax_var.append(i.split('\t')[1])
    variance.append(i.split('\t')[2])
    E_res.append(i.split('\t')[3])
    b.append(i.split('\t')[4])
for i in range(len(b)):
    b[i]=float(b[i])
    E_res[i] =float(E_res[i])
    variance[i] = float(variance[i])
    nmax_var[i] = float(nmax_var[i])
    nmax[i] = float(nmax[i])
d1 = 4.29816632e-05
d2 = -0.94831197e-03
def model_exp(x_values,params,b=0.66874030):
    """ Model function for exponential decay fit with baseline.
    For fitting data as y = c0 + c1*exp(-2*c2*L).
    Args:
        params (iterable): parameters [c0,c1,c2]
        x_values (NumPy vector): independent variable values [x0,...,x_(M-1)]
    Returns:
        (NumPy vector): values [f0,...,f_(M-1)]
    """

    (c0,c1,c2) = params

    # Debugging: Must force x_values to float.  If use x_values argument inside
    # exponential, this may come as a list of int, which, after np broadcast,
    # gives rise to a type error:
    #
    # TypeError: 'numpy.float64' object cannot be interpreted as an integer
    x_vec = np.array(x_values,dtype=float)  # force to numpy float array to avoid type error surprises
    L = ((2*(x_vec+(3/2)))**(1/2))*(b)
    K = ((2*(x_vec+(3/2)))**(1/2))*(b**(-1))
    y_values = c0 + c1*np.exp(d1*L) + c2*np.exp(d2*K**2)
    return y_values

################################################################
# residuals wrapper for least squares fit models
################################################################

def residuals(params,f_model,x_values,y_values):
    """Residual function for nonlinear fit.
    Calculates residuals d_i=y_i-f(x_i), as required for
    scipy.optimize.leastsq.  Typical signature for call will be
        fit = scipy.optimize.leastsq(residuals, c_guess, args=(f_model,x_values,y_values))
    where c_guess are the initial guesses for the parameters.
    The model function must "broadcast" over a vector of x arguments
    but need not be a full-fledged NumPy ufunc.
    Args:
        f_model (callable): model function f_model(x_values,params)
        params (NumPy vector): parameters [c0,c1]
        x_values (NumPy vector): independent variable values [x0,...,x_(M-1)]
        y_values (NumPy vector): dependent variable values [x0,...,x_(M-1)]
     Returns:
        (NumPy vector): residuals [d0,...,d_(M-1)]
    """

    d_values = y_values - f_model(x_values,params)
    return d_values

################################################################
# full wrapper for nonlinear fit
################################################################


def fit(f_model,x_values,y_values,c_guess):
    """Wrapper for scipy.optimize.leastsq least squares data fit.
    Args:
        f_model (callable): model function f_model(x_values,params)
        x_values (NumPy vector): independent variable values [x0,...,x_(M-1)]
        y_values (NumPy vector): dependent variable values [x0,...,x_(M-1)]
        c_guess (NumPy vector): initial guess for parameters [c0,c1,...]
     Returns:
        (tuple): return value from scipy.optimize.leastsq, typically 
             a tuple (params,success_code)
    """

    fit = scipy.optimize.leastsq(residuals, c_guess, args=(f_model,x_values,y_values))
    return fit

###############################################################
# data to fit
###############################################################

x_values = [2,4,6]
y_values = E_res[1:4]

def parameters(x,y,b=0.66874030):
    """Finds initial parameters for model_exp 
    of the form y = ae^bL + c
    Args:
        x (list): x values of initial three points
        y (list): y values of initial three points
    Returns:
        parameters a,b,and c
    """
    x1,x2,x3 = x
    y1,y2,y3 = y
    L1 = ((2*(x1+(3/2)))**(1/2))*(b)
    K1 = ((2*(x1+(3/2)))**(1/2))*(b**(-1))
    L2 = ((2*(x2+(3/2)))**(1/2))*(b)
    K2 = ((2*(x2+(3/2)))**(1/2))*(b**(-1))
    L3 = ((2*(x3+(3/2)))**(1/2))*(b)
    K3 = ((2*(x3+(3/2)))**(1/2))*(b**(-1))
    X1 = np.exp(d1*L1)
    X2 = np.exp(d1*L2)
    X3 = np.exp(d1*L3)
    Z1 = np.exp(d2*K1**2)
    Z2 = np.exp(d2*K2**2)
    Z3 = np.exp(d2*K3**2)
    c1 = ((y3-y2)-((Z3-Z2)*(y2-y1))/(Z2-Z1))/((X3-X2)-((Z3-Z2)*(X2-X1))/(Z2-Z1))
    c2 = ((y2-y1)-c1*(X2-X1))/(Z2-Z1)
    c0 = y1-c1*X1-c2*Z1
    return c0,c1,c2

##############################################################
#Curve fitting
##############################################################
#Plotting the data v.s the exponential fitting
curvefit1 = fit(model_exp,x_values,y_values,parameters(x=x_values,y=y_values))[0]
#Define the function of approximated curve
def exponential(x,b=0.66874030):
        L = ((2*(x+(3/2)))**(1/2))*(b)
        K= ((2*(x+(3/2)))**(1/2))*(b**(-1))
        y = (curvefit1[0]+curvefit1[1]*np.exp(d1*L)) + curvefit1[2]*np.exp(d2*K**2)
        return y
#non-discrete x values for the curve fit
x = np.arange(2,20 ,0.1)
y1 = np.zeros(len(x))
for i in range(len(x)):
    y1[i] = exponential(x[i])
#defining the output of the function y as a list
#now these are the discrete data points to plot against curve
    
nmax = [2,4,6,8,10,12,14,16,18,20]
yvals1 = E_res[1:]
plt.plot(x,y1,color = 'tab:orange',label = 'curve fit for b = 1.12468265')
plt.plot(nmax,yvals1,'o',markersize =4,color = 'tab:orange')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('infrared_03_11_50.pdf')
plt.show()