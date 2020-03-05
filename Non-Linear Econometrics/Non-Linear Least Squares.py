'''
Use root-finding algorith from scipy to solve non-linear least squares
problem.
'''

import numpy as np
import pandas as pd
import os
from scipy.optimize import root
from numpy.linalg import inv
np.random.seed(123)

T = 1000
K =3 
beta = [2,-4, 0.01]

x1 = []
x2 = []
u = []
for i in range(T):
	x1.append(np.random.uniform())
	x2.append(np.random.standard_normal())
	u.append(np.random.normal(0,0.2))
		
x2_sq = np.multiply(x2,x2)
x1dot = np.multiply(beta[1],x1)
x2dot = np.multiply(beta[2],x2_sq)

y=[]
for i in range(T):
    y.append(np.exp(beta[0] + x1dot[i]) + np.exp(x2dot[i]) + u[i])
	
def evalFunction(theta):
	beta0 = theta[0]
	beta1 = theta[1]
	beta2 = theta[2]
	
	x2_sq = np.multiply(x2,x2)
	beta1_x = np.multiply(beta1,x1)
	beta2_x = np.multiply(beta2,x2_sq)
	
	Q = []
	
	for i in range(T):
		Q.append(y[i] - np.exp(beta0 + beta1_x[i]) - np.exp(beta2_x[i]))
	
	Q_sum = float(1.0/T)*sum(Q)
	return Q_sum
		
def getGradient(theta):
	beta0 = theta[0]
	beta1 = theta[1]
	beta2 = theta[2]

	der_beta0 = []
	der_beta1 = []
	der_beta2 = []
	
	x2_sq = np.multiply(x2,x2)
	beta1_x = np.multiply(beta1,x1)
	beta2_x = np.multiply(beta2,x2_sq)
	
	for i in range(T):
		
		der_beta0.append( (y[i] - np.exp(beta0 + beta1_x[i]) - np.exp(beta2_x[i]))*(-np.exp(beta0 + beta1_x[i]))  )
		der_beta1.append( (y[i] - np.exp(beta0 + beta1_x[i]) - np.exp(beta2_x[i]))*(-np.exp(beta0 + beta1_x[i]))*x1[i] )
		der_beta2.append( (y[i] - np.exp(beta0 + beta1_x[i]) - np.exp(beta2_x[i]))*(-np.exp(beta2_x[i]))*x2_sq[i] )
		
	der_beta0_sum = round(float(sum(der_beta0)/T),12)
	der_beta1_sum = round(float(sum(der_beta1)/T),12)
	der_beta2_sum = round(float(sum(der_beta2)/T),12)

	return np.array([der_beta0_sum, der_beta1_sum, der_beta2_sum])	
	
def getMeanResid(theta):
	beta0 = theta[0]
	beta1 = theta[1]
	beta2 = theta[2]

	resids = []
	beta1_x = np.multiply(beta1,x1)
	beta2_x = np.multiply(beta2,x2_sq)

	for i in range(T):
		resids.append( (y[i] - np.exp(beta0 + beta1_x[i]) - np.exp(beta2_x[i]))**2 )
		
	omega = np.average(resids)
	return omega

def getVar(theta):
	beta0 = theta[0]
	beta1 = theta[1]
	beta2 = theta[2]
	
	x2_sq = np.multiply(x2,x2)
	beta1_x = np.multiply(beta1,x1)
	beta2_x = np.multiply(beta2,x2_sq)
	
	summand0 = []
	summand1 = []
	summand2 = []
	
	for i in range(T):
		summand0.append(np.exp(beta0 + beta1_x[i]))
		summand1.append(np.exp(beta0 + beta1_x[i])*x1[i])
		summand2.append(np.exp(beta2_x[i])*x2_sq[i])
		
	dgdb_t = np.array([summand0,summand1,summand2])
	dgdb   = np.transpose(dgdb_t)
	A_hat = (1.0/T)*np.matmul(dgdb_t,dgdb)
	B_hat = (1.0/T)*getMeanResid(theta)*np.matmul(dgdb_t,dgdb)
	A_inv = inv(A_hat)
	mult1 = np.matmul(A_inv,B_hat)
	Var = np.matmul(mult1,A_inv)
	return Var
	
def getSol(startvals):
	solns = []
	for i in range(len(startvals)):
		sol = root(getGradient, startvals[i], method='lm')
		solns.append(sol)		
	return solns
	
startingvals = [[-5.0,-5.0,-5.0],[1.0,3.0,5.0],[-2.0, 2.0,1.0],[0.0,-3.0,-1.0],[9.0,9.0,0]]
sols = getSol(startingvals)

Q_vals = [[0 for n in range(4)] for m in range(len(sols))]
Q_valdf = pd.DataFrame(Q_vals,columns=['beta0','beta1','beta2','functionval'])

for i in range(len(sols)):
    Q_valdf.ix[i,"beta0"] = sols[i].x[0]
    Q_valdf.ix[i,"beta1"] = sols[i].x[1]
    Q_valdf.ix[i,"beta2"] = sols[i].x[2]
    Q_valdf.ix[i,"functionval"] = evalFunction(sols[i].x)
    
Q_sorted = Q_valdf.sort_values(by='functionval')

beta0_hat = Q_valdf.ix[Q_valdf['functionval']==min(Q_valdf.ix[:,"functionval"]),'beta0']
beta1_hat = Q_valdf.ix[Q_valdf['functionval']==min(Q_valdf.ix[:,"functionval"]),'beta1']
beta2_hat = Q_valdf.ix[Q_valdf['functionval']==min(Q_valdf.ix[:,"functionval"]),'beta2']
theta_hat = [beta0_hat.values[0], beta1_hat.values[0], beta2_hat.values[0]]

variance = getVar(theta_hat) 
stderr_b0 = np.sqrt(variance[0,0])
stderr_b1 = np.sqrt(variance[1,1])
stderr_b2 = np.sqrt(variance[2,2])

print "Theta Hat = " + str(theta_hat)
print ""
print "Std Error of Beta 0 = " + str(stderr_b0)
print ""
print "Std Error of Beta 0 = " + str(stderr_b1)
print ""
print "Std Error of Beta 0 = " + str(stderr_b2)


