#How sensitive are retirement balances to various average market returns?
#8/11/2017

import pandas as pd 
import datetime
import matplotlib.pyplot as plt
from scipy.stats import norm 
import numpy as np

pd.options.display.float_format = '{:,.2f}'.format

#Initialize inputs:
init_cap = 15000                       #Initial savings (Capital)
annual_cont = 8000                     #Level of annual savings contributions
age_start = 30                         #Age started saving
age_end = 50                           #Age end saving
years = age_end - age_start + 1        #Total saving years
cont_trend = .02                       #Increase annual contributions by x%
N = 10                                 #Run simulation N times
rates = [.02,.03,.04,.05,.06,.07,.08]  

'''Initialize a zero matrix. This matrix will contain the cumulative saving 
   balances for all saving years except the last. The simulation will run 
   N times so there will be N columns.
'''
#kXm = yearsXN
matrix = [[0 for k in range(N)] for m in range(years)]

'''Initialize a zero matrix. This matrix will contain N ending saving balances
   for each market return value in the rates array.
'''
#jXk = N X len(rates)
cumulative = [[0 for j in range(len(rates))] for k in range(N)] 


'''The loop functions as follows:
   A market return is chosen. Then a simulation is started. A column of cumulative
   savings is generated and the final value populates the cumulative matrix.
'''
for i in range(len(rates)):
	mean = rates[i]
	stddev = .1629299
	
	for j in range(N):

		for k in range(years):
		
			if k == 0: 
				matrix[k][j] = init_cap + 0.5*annual_cont
			elif k>0 and k<years-1:
				matrix[k][j] = (1+norm.ppf(np.random.rand(), mean, stddev))*matrix[k-1][j]+annual_cont*(1+cont_trend)**(k-1)	
			else:
				cumulative[j][i] = (1+norm.ppf(np.random.rand(), mean, stddev))*matrix[k-1][j]+annual_cont*(1+cont_trend)**(k-1)

'''Next put the cumulative matrix in a DataFrame and create
   a sensitivity table that shows the deciles of cumulative
   savings given an average market return.
'''
rows = []
for i in range(N):
	rows.append(i)
	
df = pd.DataFrame(cumulative, index = rows, columns = rates)
print df	

quantiles = [.1, .2, .3, .4, .5, .6, .7, .8, .9]
summary = [[0 for k in range(len(rates))] for m in range(len(quantiles))]

'''Use the function quantile to calculate the percentiles
'''
for j in range(len(rates)):
    for i in range(len(quantiles)):
        summary[i][j] = df[rates[j]].quantile(quantiles[i])
sumdf = pd.DataFrame(summary, index = quantiles, columns = rates)
	
print sumdf





