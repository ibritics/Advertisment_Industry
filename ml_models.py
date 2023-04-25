import pandas as  pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error

##ADDING ANOTHER COMMENT ON A TEMP BRANCH
## ADDING A COMMENT ON A TEMP BRANCH
### PLEASE KEEP THE PATH CONSTANT ###
#Importing
df= pd.read_csv('C:/Users/Dell/Desktop/MODELS/Housing.csv')


#Transforming the categorical data into a binary data
def encode(val):
    if val == 'yes':
        return 1
    elif val=='no':
        return 0
    else:
        return val

#Applying before created function
df= df.applymap(encode)

#Assigning the encoder to the variable
enc= OneHotEncoder()

enc_data= pd.DataFrame(enc.fit_transform(
    df[['furnishingstatus']]).toarray())

df= df.join(enc_data)

df = df.rename(columns={0: 'furnished', 1: 'semi-furnished', 2: 'unfurnished'})
df.drop(['furnishingstatus'], axis='columns', inplace=True)

print(df)

#### ML MODELS ####

class ml_models():
    def __init__(self,df): #This part takes your global setup
        self.df= df
        self.X= self.df.iloc[:,1:len(df.columns)]
        self.y = self.df.iloc[:,0]
        #self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.3, random_state=42) for not time series
        self.X_train, self.y_train = self.df.iloc[:int(0.7 * len(df)),1:], self.df.iloc[:int(0.7 * len(df)),0] #For time series analysis because we need the continus time
        self.X_test,self.y_test= self.df.iloc[int(0.7 * len(df))+1:,1:], self.df.iloc[int(0.7 * len(df))+1:,0]
    def rf_model(n_estimators,):


### JUST TRUST TO YOUR GRID SEARCH ####
param_grid = {
    'n_estimators': [25, 50, 100, 150,500],
    'max_depth': [3, 6, 9],
    'max_leaf_nodes': [3, 6, 9],
}
grid_search = GridSearchCV(RandomForestRegressor(),
                           param_grid=param_grid)
grid_search.fit(X_train, y_train)
predict = grid_search.predict(X_test)


rmse_rf= mean_squared_error(y_test,predict,squared=False)
rmse_rf
##### YOU ARE GONNA USE IT FOR THE NEXT ####
#Adding comment as jack##