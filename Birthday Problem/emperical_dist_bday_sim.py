import pandas as pd
import numpy as np

#Grab the data from FiveThirtyEight Github
html = 'https://raw.githubusercontent.com/fivethirtyeight/data/' \
'master/births/US_births_2000-2014_SSA.csv'

df = pd.read_csv(html)

#Get rid of the leap years...this is Medium not my Ph.D.
drop_yr = ['2000','2004','2008','2012']
drop_yr_index = df[df['year'].isin(drop_yr)].index
df.drop(drop_yr_index,axis=0,inplace=True)
df.reset_index(inplace=True)

#The data doesn't have a day-of-year index so I create one 
df['year_inc'] = 0
for i in range(len(df)):
    if i==0:
        df.loc[i,'year_inc']=1
        incrementer = 1
    if i>0:
        
        if df.loc[i,'year']==df.loc[i-1,'year']:
            incrementer +=1
            df.loc[i,'year_inc'] += incrementer
        else:
            incrementer=1
            df.loc[i,'year_inc'] = 1
            
#Sum up the birth counts by the day of year var we just created
ct = df[['year_inc','births']].groupby('year_inc').sum().reset_index()
total = ct['births'].sum()
ct['prob'] = ct['births']/total

# Grab the day of year index and merge it with the count dataframe
html_doy = 'https://raw.githubusercontent.com/stevenraybon/' \
'Python/master/day_of_year.csv'
doy=pd.read_csv(html_doy)
ct = pd.merge(ct,doy,left_on='year_inc',right_on='day_of_year',how='left')

# Simulation: Not sure how many should be in cohort so I pick diff cohort 
# sizes of 20-40 people and look at the average success rates
res=[]
arr=[]
weights = ct['prob'].values
for j in range(20,40,1):
    for i in range(2000):
        samp=ct['day_of_year'].sample(j,replace=True,weights=weights).reset_index()
        
        if len(pd.unique(samp['day_of_year'])) < len(samp):
            arr.append(1)
        else:
            arr.append(0)
    res.append(np.mean(arr))
    
# Plot the results
df_res = pd.DataFrame(res,index=np.arange(20,40,1),columns=['success_rate'])
ax = df_res[df_res.index.isin([24,25,26,27,28])]
    .plot(kind='bar',legend=False)
ax.grid(True,alpha=.5)
ax.set_axisbelow(True)
ax.set_ylabel('success rate')
ax.set_xlabel('# of ppl in cohort')
title='Avg Success Rate of Sharing Birthday in Cohort of N People'
ax.set_title(title)
