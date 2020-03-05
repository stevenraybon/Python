import numpy as np
import pandas as pd
import os
from scipy.optimize import minimize
from numpy.linalg import inv
import math

data = pd.read_csv('birthweight_smoking.csv')

y = data.ix[:,"birthweight"]
xcols = ['smoker','unmarried','educ','age','drinks','nprevist','alcohol','tripre1','tripre2','tripre3','tripre0']
X = data[xcols]

sum1 = np.zeros(len(y))
sum2 = np.zeros(len(y))

def getXB(theta,i):
	beta0 = theta[0]
	beta1 = theta[1]
	beta2 = theta[2]
	beta3 = theta[3]
	beta4 = theta[4]
	beta5 = theta[5]
	beta6 = theta[6]
	beta7 = theta[7]
	beta8 = theta[8]
	beta9 = theta[9]
	beta10 = theta[10]
	
	XB = beta0 + beta1*X.ix[i,"smoker"] + beta2*X.ix[i,"unmarried"] + beta3*X.ix[i,"educ"] + beta4*X.ix[i,"age"] + 
	beta5*X.ix[i,"drinks"] + beta6*X.ix[i,"nprevist"] + beta7*X.ix[i,"alcohol"] + 
	beta8*X.ix[i,"tripre1"] + beta9*X.ix[i,"tripre2"] + beta10*X.ix[i,"tripre3"]
	
	return XB
	
#quantiles = 0.25, 0.50, 0.75
q = 0.5
def getQ(theta):
	for i in range(len(y)):
		if y[i] > getXB(theta,i):
			sum1[i] = q*np.absolute(y[i] - getXB(theta,i))
		else:
			sum2[i] = (1-q)*np.absolute(y[i] - getXB(theta,i))
			
	Q = sum(sum1)+sum(sum2)
	return Q

theta = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 
sol1 = minimize(getQ,theta,method='SLSQP')
np.set_printoptions(suppress=True)
print sol1.x


