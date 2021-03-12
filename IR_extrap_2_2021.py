# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:57:23 2021

@author: Owner
"""


import matplotlib.pyplot as plt
from numpy.linalg import inv
import numpy as np
import scipy.optimize

f = open('\\Users\\Owner\\OneDrive\\Documents\\IR_UV_data_2.txt','r')
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
    b[i]=int(b[i])
    E_res[i] =float(E_res[i])
    variance[i] = float(variance[i])
    nmax_var[i] = float(nmax_var[i])
    nmax[i] = float(nmax[i])

def model_exp(x_values,params,b=30):
    """ Model function for exponential decay fit with baseline.
    For fitting data as y = c0 + c1*exp(c2*L).
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
    y_values = c0 + c1*np.exp(c2*L)
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

def parameters(x,y,b=30):
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
    L2 = ((2*(x2+(3/2)))**(1/2))*(b)
    L3 = ((2*(x3+(3/2)))**(1/2))*(b)
    dL1 = L2-L1
    dL2 = L3-L2
    dy1 = y2-y1
    dy2 = y3-y2
    cL1 = (L1+L2)/2
    cL2 = (L3+L2)/2
    dq1 = dy1/dL1
    dq2 = dy2/dL2
    B = np.log(dq2/dq1)/(cL2-cL1)
    a1 = dy1/(np.exp(B*L2)-np.exp(B*L1))
    a2 = dy2/(np.exp(B*L3)-np.exp(B*L2))
    a = (a1+a2)/2
    c1 = y1-(a*np.exp(B*L1))
    c2 = y2-(a*np.exp(B*L2))
    c3 = y3-(a*np.exp(B*L3))
    c = (c1+c2+c3)/3
    return c,a,b

##############################################################
#Curve fitting
##############################################################
#Plotting the data v.s the exponential fitting
curvefit1 = fit(model_exp,x_values,y_values,parameters(x=x_values,y=y_values))[0]
#Define the function of approximated curve
def exponential(x):
        y = (curvefit1[1]*np.exp(curvefit1[2]*x)) + curvefit1[0]
        return y
#non-discrete x values for the curve fit
x = np.arange(2,20,0.1)
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
plt.savefig('infrared_03_11_4.pdf')
plt.show()

















