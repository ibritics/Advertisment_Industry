from sklearn.linear_model import LinearRegression
import pandas as pd
import statsmodels.api as sm #Constant/Intercept


df= pd.read_csv('C:/Users/Dell/Desktop/MODELS/Advertising.csv', index_col=0)

### LINEAR REGRESSION ###
X_train, y_train = df.copy().iloc[:int(len(df)*0.7),:-1], df.copy().iloc[:int(len(df)*0.7),-1]
X_test,y_test=  df.copy().iloc[int(len(df)*0.7)+1:,:-1],  df.copy().iloc[int(len(df)*0.7)+1:,-1]

linear_model= sm.OLS(endog= y_train, exog=X_train).fit()

linear_model.summary()


