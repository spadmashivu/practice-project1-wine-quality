#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
import warnings
warnings.filterwarnings('ignore')


# In[17]:


df=pd.read_csv("wine.csv")
df


# In[18]:


df.shape


# In[26]:


df.info()


# In[27]:


df.describe()


# In[33]:


training_data=df.iloc[:1599]


# In[34]:


training_features_data=training_data.drop('quality', axis='columns')
training_labels_data=training_data['quality'].copy()


# In[122]:


testing_data=df.iloc[1599:]


# In[123]:


testing_features_data=testing_data.drop('quality', axis='columns')
testing_labels_data=testing_data['quality'].copy()


# In[124]:


training_data


# In[ ]:





# In[125]:


from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


# In[126]:


my_pipeline=Pipeline([
    ('imputer', SimpleImputer()),
    ('scaler', StandardScaler())
])
x_train=my_pipeline.fit_transform(training_features_data)

x_test=my_pipeline.fit_transform(testing_features_data)


# In[111]:


correlation=df.corr()
correlation['quality'].sort_values(ascending=False)


# Using Linear model (Model and k-fold cross validation)
# 

# In[117]:


from sklearn.linear_model import LinearRegression


# In[118]:


model=LinearRegression()
model.fit(x_train, training_labels_data)


# In[119]:


x =model.predict(x-test)


# In[57]:


from sklearn.model_selection import cross_val_score
scores=cross_val_score(model, training_features_data, training_labels_data, scoring="neg_mean_squared_error", cv=10)
rmse_scores=np.sqrt(-scores)
len(list(rmse_scores))


# In[58]:


list(rmse_scores)


# In[59]:


def print_scores_of_validation(scores):
    print("Scores:", scores)
    print("mean:", scores.mean())
    print("standard deviation:", scores.std())


# In[60]:


print_scores_of_validation(rmse_scores)


# In[66]:


from sklearn.metrics import mean_squared_error
lin_mse=mean_squared_error(testing_labels_data, x)
lin_rmse=np.sqrt(lin_mse)


# In[67]:


lin_mse, lin_rmse


# Using RandomForest (Model and k-fold cross validation)
# 

# In[68]:


from sklearn.ensemble import RandomForestRegressor
model2=RandomForestRegressor()
model2.fit(x_train, training_labels_data)


# In[69]:


# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 10, stop = 1200, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt', 'log2']
# Maximum number of levels in tree
max_depth = [2,150]
# Minimum number of samples required to split a node
min_samples_split = [2, 5]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2]
# Method of selecting samples for training each tree
bootstrap = [True, False]


# In[70]:


# Create the param grid
param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)


# In[71]:


from sklearn.model_selection import RandomizedSearchCV
rf_RandomGrid = RandomizedSearchCV(estimator = model2, param_distributions = param_grid, cv = 100, verbose=2, n_jobs = 4)


# In[73]:


rf_RandomGrid.fit(x_train, training_labels_data)


# In[74]:


rf_RandomGrid.best_params_


# In[75]:


print (f'Train Accuracy - : {rf_RandomGrid.score(x_train,training_labels_data):.3f}')
print (f'Test Accuracy - : {rf_RandomGrid.score(x_test,testing_labels_data):.3f}')


# In[76]:


x_=model2.predict(x_test)
x2=np.round(x_)


# In[82]:


scores2=cross_val_score(model2, training_features_data, training_labels_data, scoring="neg_mean_squared_error", cv=10)
rmse_scores2=np.sqrt(-scores2)


# In[83]:


print_scores_of_validation(rmse_scores2)


# In[84]:


lin_mse2=mean_squared_error(testing_labels_data, x2)
lin_rmse2=np.sqrt(lin_mse2)


# In[85]:


lin_mse2, lin_rmse2


# In[86]:


testing_data.to_csv('Testing and predicted data.csv')


# In[87]:


df2=pd.read_csv('Testing and predicted data.csv')


# In[88]:


df2["Linear_regression"]=x
df2["Decision_Tree"]=x1
df2["Random_Forest"]=x2


# In[89]:


df2.to_csv('Testing and predicted data.csv')


# In[90]:


def predict_quality(parameters): #enter space seperated 11 integer values as parameters
        
        x=parameters.split()
        arr=np.array(x)

        z=np.reshape(arr, (1,-1))
        y=my_pipeline.fit_transform(z)
        output=model2.predict(y)
        if output<6:
            print("Bad")
        elif output==6:
            print("Average")
        else:
            print("Good")
    


# In[91]:


df2


# In[92]:


truth_quality=df2['quality']


# In[93]:


truth_label=[]
for values in truth_quality:
    if values>6:
        truth_label.append('good')
    elif values<6:
        truth_label.append('bad')
    else:
        truth_label.append('average')
        


# In[94]:


df2['truth_label']=truth_label


# In[95]:


df2


# In[96]:


rf_label=[]
Random_Forest=df2['Random_Forest']
for entries in Random_Forest:
    if entries>6:
        rf_label.append('good')
    elif entries<6:
        rf_label.append('bad')
    else:
        rf_label.append('average')


# In[98]:


df2['rf_label']=rf_label
df2.to_csv('Testing and predicted data.csv')
i=0
for a,b in zip(truth_label,rf_label):
    if a==b:
        i=i+1
i


# In[ ]:




