#OLS and Two Stage Least Squares Simulation Results
#Simulation follows Steve Berry (1994)
'''
Neglecting unobserved product characteristics in regression of mean utility 
on price and observed product characteristics leads to biased coefficients. 
Two Stage Least Squares is used to correct for this bias. The problem stems from
unobserved product characteristics being correlated with prices - a feature of 
the supply-side model: products with more desirable unobserved characteristics
are assumed to have a higher marginal cost of production and hence are correlated 
with a higher price. 
'''

from numpy import random as rand
import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
%matplotlib inline

#Fixes the random number generator so results can be replicated
rand.seed(123)

#Declare variables of interest
beta_0  = 5
beta_x  = 2
alpha   = 1
gamma_0 = 1
gamma_x = 0.5
gamma_w = 0.25
sigma_omega = 0.25
sigma_d = 0.25
N = 1000000 #Consumers
markets = 500
duopolists = 2

###Create exogenous data table

#Create column for Markets
markcols = []
for i in range(markets):
    markcols.append(i+1)
    markcols.append(i+1)

#Create column for duopolists
duopcols = []
for i in range(markets):
	duopcols.append(1)
	duopcols.append(2)
	
#Create matrix so random entries can be used
#mXk matrix 
matrix = [[0 for k in range(4)] for m in range(markets*2)]
 
#Initialize with N(0,1) draws
for i in range(markets*2):
	for j in range(4):
		matrix[i][j] = rand.standard_normal()

#Turn vecotrs and matrix into DataFrame for further manipulation down below
#Final table to be exported is exog_matrix
markcols_df = pd.DataFrame(markcols)
duopcols_df = pd.DataFrame(duopcols)
cols_df = pd.concat([markcols_df, duopcols_df], axis=1)
matrix_df = pd.DataFrame(matrix)
exog_matrix = pd.concat([cols_df, matrix_df], axis=1)
exog_matrix.columns = ['mkt','firm','x', 'w', 'derr', 'cerr']
	
#Export the exogenous data to a csv file
#exog_matrix.to_csv('exog_data.csv', sep=',', index=False)

#Functions: Below is a series of functions that will be used later in the program
#The first four relate to computing market shares
#The last two relate to calculating marginal cost

#Market shares are calculated using s_j = exp{Xb + ap}/sum[exp{Xb + ap}]
#GetDenom obtains the denominator of the market share formula
def getDenom(p1,p2,i):
		
	denom = 1 + np.exp(beta_0 + beta_x*exog_matrix.ix[i,"x"] + alpha*p1 + exog_matrix.ix[i,"derr"]) + \
	np.exp(beta_0 + beta_x*exog_matrix.ix[i+1,"x"] + alpha*p2 + exog_matrix.ix[i+1,"derr"])
	
	return denom

#getShare obtains the numerator of the logit market share formula and then calls getDenom
#to get the denominator. 
def getShare1(p1,p2,i):
	num1 = float(np.exp(beta_0 + beta_x*exog_matrix.ix[i,"x"] + alpha*p1 + exog_matrix.ix[i,"derr"] ))
	return num1/getDenom(p1,p2,i)
	
def getShare2(p1,p2,i):
	num2 = float(np.exp(beta_0 + beta_x*exog_matrix.ix[i+1,"x"] + alpha*p2 + exog_matrix.ix[i+1,"derr"] ))
	return num2/getDenom(p1,p2,i)

#getCost calculates the marginal cost using prices and exogenous data
def getCost1(j):
	return np.exp(gamma_0 + gamma_x*exog_matrix.ix[j,"x"] + sigma_d*exog_matrix.ix[j,"derr"] + \
	gamma_w*exog_matrix.ix[j,"w"] + sigma_omega*exog_matrix.ix[j,"cerr"])
	
def getCost2(j):
	return np.exp(gamma_0 + gamma_x*exog_matrix.ix[j+1,"x"] + sigma_d*exog_matrix.ix[j+1,"derr"] + \
	gamma_w*exog_matrix.ix[j+1,"w"] + sigma_omega*exog_matrix.ix[j+1,"cerr"])
	
#Price Convergence 
#Make initial guess about prices and then iterate until they converge to value that 
#makes FOC close to zero. Closeness is described by tolerance level
eqPrice = []
maxPrice1 = 40
maxPrice2 = 40
price1 = 0
price2 = 0

#Initializing a vector of initial guess of 0 for each firm for all 500 markets
for i in range(markets*2):
	eqPrice.append(0)

tol = .4 #tolerance level
increment = .1

#Iterate through each market
for i in range(0, markets*2, 2):
	
	while True: 
		diff1 = price1 - 1/(1-getShare1(price1,price2,i)) - getCost1(i) #FOC: needs to be approx equal to 0		
		if np.absolute(diff1) < tol:
			eqPrice[i] = price1
			break
			
		else:
			#grid search over values of price1 and price2 
			if price1 >= maxPrice1:
			
				if price2 >= maxPrice2:
					print "In market " + str(i) + ": no eqm price was found"
				else:
					price1 = 0
					price2 += increment
			else:
				price1 += increment

				
	while True: 
		diff2 = price2 - 1/(1-getShare2(price1,price2,i)) - getCost2(i) #FOC: needs to be approx equal to 0
		
		if np.absolute(diff2) < tol:
			eqPrice[i+1] = price2
			break
			
		else:
			#grid search over values of price1 and price2 
			if price2 >= maxPrice2:
			
				if price1 >= maxPrice1:
					print "In market " + str(i) + ": no eqm price was found"
				else:
					price2 = 0
					price1 += increment
			else:
				price2 += increment
        
#Define series of vectors to create matrix for OLS
#OLS will run model delta(i) = ln(share(i)/outside_share(i)) = f(x(i), eqPrice(i))
#Two Stage Least Squares (2SLS) will run similar model but will instrument 
#other-firm demand characteristics and supply characteristics, w(i)

eqQuant = []       #equilibrium market quantity
eqShares = []      #equilibrium market share
outside_share = [] #equilibrium share of outside good
delta = []         #mean utility
cross_char = []    #other-firm demand characteristics
tot_share = []     #total market share of 2 "inside" goods

for i in range(markets*2):
    eqQuant.append(i)
    eqShares.append(i)
    outside_share.append(i)
    delta.append(i)
    cross_char.append(i)
    tot_share.append(i)

for i in range(0,markets*2,2):
    eqQuant[i] = round(getShare1(eqPrice[i],eqPrice[i+1],i)*N,0)
    eqQuant[i+1] = round(getShare2(eqPrice[i],eqPrice[i+1],i)*N,0)

    eqShares[i] = getShare1(eqPrice[i],eqPrice[i+1],i)
    eqShares[i+1] = getShare2(eqPrice[i],eqPrice[i+1],i)
    
    outside_share[i] = (1 - eqShares[i] - eqShares[i+1])
    outside_share[i+1] = (1 - eqShares[i] - eqShares[i+1])
    
    delta[i] = np.log(eqShares[i]/outside_share[i])
    delta[i+1] = np.log(eqShares[i+1]/outside_share[i+1])
    
    cross_char[i] = exog_matrix.ix[i+1,"x"]
    cross_char[i+1] = exog_matrix.ix[i,"x"]
    
    tot_share[i] = round(eqShares[i] + eqShares[i+1],6)
    tot_share[i+1] = round(eqShares[i] + eqShares[i+1], 6)
    
    
#Turn above arrays into DataFrames for ease of use
#final product: matrix of exogenous/endogenous observations - obs_matrix
eqPrice_df = pd.DataFrame(eqPrice)
eqQuant_df = pd.DataFrame(eqQuant)
eqShares_df = pd.DataFrame(eqShares)
delta_df   = pd.DataFrame(delta)
cross_char_df = pd.DataFrame(cross_char)
outside_share_df = pd.DataFrame(outside_share)
tot_share_df = pd.DataFrame(tot_share)
eqm_data = pd.concat([eqQuant_df, eqPrice_df, eqShares_df, tot_share_df, outside_share_df, delta_df, cross_char_df], axis=1)
colnames = ("mkt", "firm", "x", "w")
obs_matrix = pd.concat([exog_matrix.ix[:,colnames], eqm_data], axis=1)
obs_matrix.columns = ['mkt','firm','x', 'w', "quantity", "price", "shares", "totalShare", "outsideShare", "delta", "crosschar"]    

#Create a new matrix - obs_matrix2 - that contains all the information in obs_matrix except data points for which the total
#market share is equal to 1
obs_matrix2 = obs_matrix[obs_matrix.ix[:,"totalShare"]!=1]

#create new matrix that reindexes so that concatenation works later
new_obs = obs_matrix2.reset_index(drop=True)

#First model: OLS
#regress delta on x and price
y = obs_matrix2.ix[:,"delta"]
X_cols = ["x", "price"]
X = new_obs.ix[:,X_cols]
model = linear_model.LinearRegression()
ols_model = model.fit(X,y)
ols_pred = ols_model.predict(X)

#Plot delta on price and plot predicted values
print "OLS coefficients are: intercept = " + str(round(ols_model.intercept_,2)) + "; slope_x = " + \
str(round(ols_model.coef_[0],2)) + "; slope_p = " + str(round(ols_model.coef_[1],2))

#Two Stage Least Squares
#(1) First Stage: regress price on x and w (demand and cost characteristics)
model2SLS_1 = linear_model.LinearRegression()
X_FS_cols = ["x","w", "crosschar"]
X_FS = new_obs.ix[:,X_FS_cols]
y_FS = new_obs.ix[:,"price"]
model_FS = model2SLS_1.fit(X_FS,y_FS)
model_FS_pred = model_FS.predict(X_FS)
FS_pred_df = pd.DataFrame(model_FS_pred)

#Two Stage Least Squares
#(2) Second Stage: regress delta on x and p_hat
model2SLS_2 = linear_model.LinearRegression()
model_FS_pred_df = pd.DataFrame(model_FS_pred)
X_SS = pd.concat([new_obs.ix[:,"x"],model_FS_pred_df], axis=1)
y_SS = new_obs.ix[:,"delta"]
SS_model = model2SLS_2.fit(X_SS, y_SS)
SS_model_pred = SS_model.predict(X_SS)

#Plot OLS with 2SLS to compare fit
print "OLS coefficients are: intercept = " + str(round(ols_model.intercept_,2)) + "; slope_x = " + \
str(round(ols_model.coef_[0],2)) + "; slope_p = " + str(round(ols_model.coef_[1],2))

print "2SLS coefficients are: intercept = " + str(round(SS_model.intercept_,2)) + "; slope_x = " + \
str(round(SS_model.coef_[0],2)) + "; slope_p = " + str(round(SS_model.coef_[1],2))
