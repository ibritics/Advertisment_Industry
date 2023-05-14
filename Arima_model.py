#### ARIMA ####

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA
import statsmodels as sm

#### Loading DF ###
df= pd.read_csv('C:/Users/Dell/Desktop/MODELS/Advertising.csv', index_col=0)

df.head()

df['sales'].describe()


#Creating the sequence of days 

# Define start date
start_date = datetime(2021, 11, 26)

# Create list of dates
dates = [start_date + timedelta(days=i) for i in range(200)]

# Format dates as strings
formatted_dates = [date.strftime('%Y-%m-%d') for date in dates]

# Print list of dates
print(formatted_dates)

df['dates']= formatted_dates
df= df.set_index('dates')


## ARIMA ##
#ARIMA(20,0,20) 

print(round(adfuller(df['sales'])[1],4))

plot_pacf(df['sales'])
plt.show()

from statsmodels.tsa.stattools import pacf
import numpy as np
pacf_values = pacf(df['sales'])

significant_lags = np.where(np.abs(pacf_values) > 1.96 / np.sqrt(len(df['sales'])))[0]

P= max(significant_lags)

print(P)

plot_acf(df['sales'])
plt.show()


from datetime import datetime, timedelta

"""
We are creating a date values to use as an index
"""
start_date = datetime(2021, 1, 1)
weeks_to_add = 200

dates = [start_date + timedelta(weeks=i) for i in range(weeks_to_add)]

df['dates']=dates #We are assigning the date 

df = df.set_index('dates') #We are setting the date as an index

# split df into input and output columns
X_train, y_train = df.copy().iloc[:int(len(df)*0.7),:-1], df.copy().iloc[:int(len(df)*0.7),-1]
X_test,y_test=  df.copy().iloc[int(len(df)*0.7)+1:,:-1],  df.copy().iloc[int(len(df)*0.7)+1:,-1]

y_train.head()
import statsmodels.api as sm
sm.tsa.ARIMA()

arima_model= sm.tsa.ARIMA(y_train,order=(20,0,20)).fit()

print(arima_model.summary())

df


#### PART TWO###
####PCA ####
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# Separate the X regressors and Y regressant
X = df.drop('sales', axis=1)
Y = df['sales']

# Standardize the X df
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Instantiate a PCA object and fit it to the df
pca = PCA(n_components=len(X.columns))
pca.fit(X_std)

# Transform the df into the new lower-dimensional space
X_pca = pca.transform(X_std)

# Print the explained variance ratios for each principal component
print('Explained variance ratios:', pca.explained_variance_ratio_)

# Print the original shape of the df and the shape of the transformed df
print('Original shape:', X.shape)
print('Transformed shape:', X_pca.shape)

# Create a horizontal bar plot of the explained variance ratios for each principal component with variable labels
fig, ax = plt.subplots()
ax.barh(range(len(pca.explained_variance_ratio_)), pca.explained_variance_ratio_)
ax.set_xlabel('Explained Variance Ratio')
ax.set_ylabel('Principal Component')

# Add labels for the variables
ax.set_yticks(range(len(X.columns)))
ax.set_yticklabels(X.columns)

plt.show()