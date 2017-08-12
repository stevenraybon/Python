#MC Call Valuation
#8/11/2017

#Draw from N(0,1) N times
#Have a starting stock price
#Assume a growth rate, have a time frame, assume no dividends, assume interest rate
#generate a set of N time T stock prices
#calculate the payoff (assume strike)
#find average payoff
#discount it to the present

from numpy import random
from numpy import exp
from numpy import sqrt
from numpy import power
from numpy import maximum
from numpy import average
from matplotlib import pyplot as plt

random.seed(123)

s_0   = 50
K     = 60 
sigma = .2
alpha = .03
r     = .05
T     = 1
N     = 1000
M     = 1000

'''getCallPrice(s_0, K, sigma, alpha, r, T, N=1000, M=1000) 
   s_0   = initial stock price
   s_T   = stock price at time T
   sigma = historical vol of S
   alpha = ___
   r     = discounting interest rate
   T     = option duration 
   N     = # of s_T simulations 
   M     = # of options simulated      
'''
def getCallPrice(s_0, K, sigma, alpha, r, T, N=1000, M=1000):

	prices = []
	payoff = []
	call_prices = []

	for j in range(M):
		
		for i in range(N):

			z = random.normal(0,1)
			s_T = (s_0)*exp((alpha - (.5)*power(sigma,2))*T + sqrt(T)*sigma*z)
			prices.append(s_T)
			payoff.append(maximum(0,s_T - K))

		#Determine the average payoff
		avg_payoff = average(payoff)

		#discount the average payoff
		call_price = exp(-r*T)*avg_payoff
		call_prices.append(call_price)
		
	plt.plot(call_prices, linewidth=2.0)
	plt.xlabel('# of Simulations')
	plt.ylabel('Simulated Call Value')
	plt.title('MC Call Valuation')
	plt.grid(True)
	plt.show()
		
getCallPrice(s_0, K, sigma, alpha, r, T)	
	
