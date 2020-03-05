###Comparing Imputation Methods
###10/10/2017

'''
Goal of code:
Want to compare "prediction errors" of different imputation methods.

Will compare Scikit-Learn Imputation methods (mean, median, mode) and 
a method using OLS to predict missing values.

Going forward, a better way to isolate predictive power 
is to divide sample into 'units' and then find means across diff units.

Sections: 
(1) Get data: I used data from NLSY (longitudinal study) that I had on hand.
(2) Clean data: get rid of NaN's in dataset
(3) Remove values of annual earnings randomly
(4) Use different imputation methods to compute imputed values of missing data
(5) Compare sum of squared "error" of predicted values vs actual values
'''

from numpy import random as rand
import pandas as pd
import numpy as np
import os
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.preprocessing import Imputer
import math

'''
(1) Get Data
'''
df = pd.read_csv('NLSY97.csv')

col_keep = ['inid', 'year', 'age', 'married', 'female', 'urban', 'hrsusu', 'ann_earnings', 'edulev', 'White', 'Black', 'Am_Indian', 'Asian']
data = df[col_keep]

'''
(2) Clean Data
There are lots of NaN's:
Create a new dataframe that gets rid of any records with NaN's ... 
present in any column 
'''
#data[data['age'].isnull()==True].head(10)
#data.isnull().values.any()
cleandata_int = data.dropna(axis=0, how="any")
#cleandata_int.isnull().values.any()

'''
Generate dummy variables for vars that are entered as factors.
Then drop the old columns to keep DF tidy.
Merge cleandata_int with the dummy vars to get cleanData
'''
sex_d = pd.get_dummies(cleandata_int['female'])
edulev_d = pd.get_dummies(cleandata_int['edulev'])

cols_to_drop = ['edulev', 'female']
cleandata_int.drop(cols_to_drop, axis=1, inplace=True)

to_merge = [cleandata_int, sex_d, edulev_d]
cleanData = pd.concat(to_merge, axis=1)

'''
(3) Remove Values of Earnings Randomly

Create a random vector
Based on the random values in it, we can remove x% of data
by indicating a threshold (i.e. if we want to remove 10% 
of the data, then we can specify data be removed for random
numbers over 0.90 .
'''
rand.seed(123)
randvec = []
for i in range(len(cleanData)):
    randvec.append(rand.uniform(0,1))

'''
We will just randomly remove values of annual earnings. This 
way, we can use regression and the other variables to come up 
with a model to predict the values of the missing entries. 

For about 10% of the data, replace ann_earnings with NaN
keep the real values in the missing_earn array
this will just keep an index corresponding to the data removed
'''
missing_earn = [[0 for n in range(2)] for m in range(len(cleanData))]
truncdata = cleanData.copy()

for i in range(len(cleanData)):
    if randvec[i]>.9 :
        missing_earn[i][0] = i
        missing_earn[i][1] = cleanData.iloc[i,6]
        truncdata.iloc[i,6] = np.nan
    else:
        missing_earn[i][0] = i
        missing_earn[i][1] = np.nan

missing_earn_df = pd.DataFrame(missing_earn, columns=['index','value'])
missing = missing_earn_df.dropna()

'''
(4) Imputation Methods
Here we use the 3 imputation methods in Scikit-Learn's Imputer() class
-Mean
-Median
-Mode

I also specify a linear regression model that uses all other variables
as covariates to predict the values of the missing entries

For each imputer, the data is taken from the truncated data matrix (truncdata)
and the imputer is applied to the raw data. The data is then transformed into
a pandas dataframe. 
'''
imputer_mean = Imputer(strategy='mean')
imputer_median = Imputer(strategy='median')
imputer_mode = Imputer(strategy='most_frequent')

cols = truncdata.columns
truncdata_values = truncdata.values

'''
Mean
'''
transformed_values_mean = imputer_mean.fit_transform(truncdata_values)
data_mean_df = pd.DataFrame(transformed_values_mean,columns=cols)

'''
Median
'''
transformed_values_median = imputer_median.fit_transform(truncdata_values)
data_median_df = pd.DataFrame(transformed_values_median,columns=cols)

'''
Mode
'''
transformed_values_mode = imputer_mode.fit_transform(truncdata_values)
data_mode_df = pd.DataFrame(transformed_values_mode,columns=cols)

'''
Regression
'''

'''
Need to define X and y"
X will contain the regression covariates and records that do 
not correspond to NaN values for annual earnings. truncdata_no_nan
is thus created and y,X created from it. 
'''
cols_drop = ['inid', 'year', 'female', 'HighSchool','ann_earnings']
truncdata_no_na = truncdata.dropna()
X_train = truncdata_no_na.drop(cols_drop, axis=1, inplace=False)
y_train = truncdata_no_na["ann_earnings"]

'''
Train/Test model:

The test values for X are the NaN values from the truncdata matrix. The ols_model
contains the model estimates. We will use those estimates to predict values of 
annual earnings using the covariate values corresponding to NaN's. 

Lastly, turn results into dataframe.
'''
model = linear_model.LinearRegression()
ols_model = model.fit(X_train,y_train)
X_test = truncdata.ix[truncdata["ann_earnings"].isnull()==True,X_train.columns]
ols_predicted  = ols_model.predict(X_test)
pred_df = pd.DataFrame(ols_predicted, index=missing["index"])
pred_df.columns = ["OLS_Pred"]

'''
Use predicted OLS values to impute.

Create indices in truncdata (the df that contains
all values - NaN and non-NaN - of annual earnings) and 
in pred_df (the df that contains predicted values)
so that a new matrix can be created that merges the two
dataframes. The merging will allow us to get the predicted 
values in with the actual values and then we can create a column
that combines both. 

Final result is a df that includes the actual and imputed values 
of annual earnings. Call it imputedData.
'''
pred_df["indexes"]=missing["index"]
index_copy = []
for i in range(len(truncdata)):
    index_copy.append(i)
	
truncdata["indexes"] = index_copy
merged = pd.merge(truncdata, pred_df, on='indexes', how='left')

'''
If annual earnings is NaN (has been removed) then populate the
new_earnings column with imputed values. If it's not NaN, then 
populate column with actual value. 
'''
for i in range(len(merged)):
    if math.isnan(merged["ann_earnings"][i]):
        merged.ix[i,"new_earnings"] = merged.ix[i,"OLS_Pred"]
    else:
        merged.ix[i,"new_earnings"] = merged.ix[i,"ann_earnings"]

'''
Collect results into clean dataframe
'''
cleanedData_act = cleanData["ann_earnings"].reset_index()
compare_df = [cleanedData_act["ann_earnings"], data_mean_df["ann_earnings"], data_median_df["ann_earnings"], data_mode_df["ann_earnings"], merged["new_earnings"]]
imputedData = pd.concat(compare_df, axis=1)
imputedData.columns = ["Actual", "Impute(Mean)", "Impute(Median)", "Impute(Mode)", "OLS"] 

'''
(5) Compare prediction error
Create a dataframe that includes sqrt[(predicted - actual)^2] for 
each imputation method. About 90% of the entries will be 0 since
the data didn't change. 

Sum up the errors and put into an array that 
will be used to graph.
'''
pred_error_cols = ['Imputed Mean', 'Imputed Median', 'Imputed Mode', 'Imputed OLS']
pred_error_data = [[0 for n in range(4)] for m in range(len(imputedData))]
pred_error_df = pd.DataFrame(pred_error_data,columns=pred_error_cols)

for i in range(len(imputedData)):
    pred_error_df.ix[i,'Imputed Mean'] = math.sqrt((imputedData.ix[i,"Impute(Mean)"] - imputedData.ix[i,"Actual"] )**2)
    pred_error_df.ix[i,'Imputed Median'] = math.sqrt((imputedData.ix[i,"Impute(Median)"] - imputedData.ix[i,"Actual"] )**2)
    pred_error_df.ix[i,'Imputed Mode'] = math.sqrt((imputedData.ix[i,"Impute(Mode)"] - imputedData.ix[i,"Actual"] )**2)
    pred_error_df.ix[i,'Imputed OLS'] = math.sqrt((imputedData.ix[i,"OLS"] - imputedData.ix[i,"Actual"] )**2)

'''
Calculate sums
'''
sum_mean = pred_error_df['Imputed Mean'].sum()
sum_median = pred_error_df['Imputed Median'].sum()
sum_mode = pred_error_df['Imputed Mode'].sum()
sum_ols = pred_error_df['Imputed OLS'].sum()

sums = [sum_mean, sum_median, sum_mode, sum_ols]
sum_arr = np.array(sums)

'''
Graph Results
'''
tick_labels = ('Mean', 'Median', 'Mode', 'OLS')
heights = [sum_mean, sum_median, sum_mode, sum_ols]
x = range(len(sum_arr))
plt.bar(x,heights, align='center', alpha=0.5 )
plt.xticks(range(len(sum_arr)), tick_labels)
plt.title("Prediction Error by Imputation Method")
plt.savefig('PredErrorBar.pdf')
plt.show()
