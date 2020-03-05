'''
Simple Softmax (Multinomial Logit) Model using weather data to train 
supervised model to predict corresponding location.

3 Locations used: Chapel Hill (KIGX), Houston (KMCJ), San Fran (KSFO)
This code builds/trains the model and then passes an out-of-sample record 
to predict. 

The data is collected in a sqlite3 DB
'''

import os
import sqlite3
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

'''
Use this to obtain all the data from the weather table
'''
def getAllData():
	conn = sqlite3.connect('weather.db')
	c=conn.cursor()
	
	#get data
	c.execute('''SELECT * FROM historical_weather''')
	data = c.fetchall()
	cols = ["key", "time", "date", "temp", "heatindex", "dewpoint","humidity", "pressure", "vis"]
	df = pd.DataFrame(data, columns=cols)
	conn.commit()
	conn.close()
	
	return df
	
'''
The data that's pulled will have strings (the Key var)
and we want to convert those to integer labels for 
the SKL algorithm.

the array that's passed is a dataframe
'''
def getLabel(vec):
    data = pd.DataFrame(vec[:],columns=['key'])
    num = len(data['key'].unique())
    keys = data['key'].unique()

    '''
    Loop: create a copy of the input vector 
    '''
    data['label'] = np.nan
    for i in range(len(data)):
        for j in range(num):
            if data.iloc[i,0] == keys[j]:
                data.ix[i,"label"] = j

    y_lab = data['label']
    return y_lab

	
'''
After the algorithm runs we will have results by label (not key)
so this translates the labels back in to the keys from the original data
'''
def getKeys(y_key, y_lab, y_res):
	uniq_key = pd.DataFrame(y_key[:].unique(), columns=['uniquekeys'])
	uniq_lab = pd.DataFrame(y_lab[:].unique(), columns=['uniquelabs'])
	y_comb = pd.concat([uniq_key, uniq_lab], axis=1)
    
	for i in range(len(y_comb)):
		if y_res == y_comb.ix[i,"uniquelabs"]:
			result = y_comb.ix[i,"uniquekeys"]
			return result
	
'''
Construct model and make prediction:
For now, use only numerical variables

First, create df that pulls all the data in the Weather DB
Then generate factor variable array from the location keys coming out of the DB
'''
df = getAllData()
y_key = df['key']
y = getLabel(y_key)
Xcols = ['temp', 'heatindex', 'dewpoint', 'humidity', 'pressure', 'vis']
X1 = df.ix[:,Xcols]

softmax = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
model = softmax.fit(X1,y)

'''
Pulled this data from Weather Underground to make an out of sample prediction
Chapel Hill (Key=KIGX), 12:56 AM, 10/16/17
'''
new_rec = [70.0, 0.0, 64.9, 84, 29.99, 10.00] 

#using .predict predicts the specific class
pred = softmax.predict([new_rec])
 
#using .predict_proba calculates the associated probabilities of being in each class
pred_prob = softmax.predict_proba([new_rec])

print "predicted probabilities = " + str(pred_prob) #Predicted Probabilities = (0.4566, 0.2698, 0.2736)

print "Predicted Key = " + str(pred) #Predicted Key = KIGX (This is correct!)
