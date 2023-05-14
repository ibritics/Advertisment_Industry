import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 

df= pd.read_csv('C:/Users/Dell/Desktop/MODELS/Advertising.csv', index_col=0)

df.head()


### CORRELATION MATRIX ####
corr_matrix= df.corr()
fig= plt.subplots(figsize=(10,10))
sns.heatmap(corr_matrix, annot= True, cmap='coolwarm')
plt.show()

### PAIRPLOT TO FIND THE RELATION BETWEEN VARIABLES ###
sns.pairplot(df,x_vars=['TV','radio','newspaper'],y_vars=['sales'])
plt.show()