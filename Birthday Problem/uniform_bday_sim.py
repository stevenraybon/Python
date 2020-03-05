import pandas as pd
import numpy as np

html_doy = 'https://raw.githubusercontent.com/stevenraybon/' \
'Python/master/day_of_year.csv'

doy=pd.read_csv(html_doy)

# Simulation: take 23 random bdays. If there are duplicates
# the unique bday counts will be shorter in length then the
# the original array

arr=[]
avg=[]
K=5000
for i in range(K):
    samp=doy['day_of_year'].sample(23,replace=True).reset_index()
    if len(pd.unique(samp['day_of_year'])) < len(samp):
        arr.append(1)
        avg.append(np.mean(arr))
    else:
        arr.append(0)
        avg.append(np.mean(arr))

#Plot the results
pl = pd.DataFrame(avg)
ax = pl.plot(ylim=[.3,.7],legend=False)
ax.set_xlabel('iterations')
ax.set_ylabel('probability')
ax.set_title('Average Success Rate After K Iterations')
ax.grid(True)
