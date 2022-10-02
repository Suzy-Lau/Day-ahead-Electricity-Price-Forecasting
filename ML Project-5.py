#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


energy = pd.read_csv('energy_dataset.csv')


# In[3]:


weather = pd.read_csv("weather_features.csv")


# In[4]:


energy.head()


# In[5]:


energy.dtypes


# In[6]:


energy.isnull().sum()


# In[7]:


energy.describe().T


# In[8]:


energy.hist(figsize=(25, 30), bins=50, xlabelsize=10, ylabelsize=10)
plt.show()


# In[9]:


cols = ['generation fossil peat','generation marine', 'generation geothermal','generation hydro pumped storage aggregated','generation wind offshore','generation fossil oil shale',
        'forecast wind offshore eday ahead',
        'generation fossil coal-derived gas']
energy = energy.drop(columns=cols, axis=1)


# In[10]:


energy = energy.drop('price day ahead', axis=1)


# In[11]:


import seaborn as sns


# In[12]:


plt.figure(figsize=(15,10))
sns.histplot(energy,x='price actual');


# In[13]:


weather.dtypes


# In[14]:


weather.isnull().sum()


# In[15]:


weather.describe(include='all').T


# In[16]:


weather.hist(figsize=(10, 10), bins=50, xlabelsize=10, ylabelsize=10)
plt.show()


# In[17]:


col_zero = ['pressure','rain_3h','snow_3h']
weather = weather.drop(columns=col_zero, axis=1)


# ### Descriptive analysis for time series

# In[18]:


energy['time'] = pd.to_datetime(energy['time'], utc=True, infer_datetime_format=True)
energy = energy.set_index('time')


# In[19]:


price_actual = energy['price actual']


# In[20]:


price_actual.plot(style="-o", figsize=(10, 5));


# In[21]:


price_actual[:200].plot(style="-o",figsize=(10, 5))
plt.ylabel("Electricity Price")
plt.title("Electricity Price versus Time(First 200 samples)", fontsize=16)
plt.legend(fontsize=14);


# In[22]:


monthly_electric_price = energy['price actual'].asfreq('M')


# In[23]:


monthly_electric_price.plot(style="-o",figsize=(10, 7))
plt.ylabel("Monthly Electricity Price")
plt.title("Monthly Electricity Price from 31 Dec 2014 to 31 Dec 2018", fontsize=16)
plt.legend(fontsize=14);


# In[24]:


conda install statsmodels


# In[25]:


from statsmodels.tsa.seasonal import seasonal_decompose


# In[26]:


y=monthly_electric_price


# In[27]:


decomposition = seasonal_decompose(monthly_electric_price)

plt.plot(y, label = 'Original')
plt.legend(loc = 'best')

trend = decomposition.trend
plt.show()
plt.plot(trend, label = 'Trend')
plt.legend(loc = 'best')

seasonal = decomposition.seasonal
plt.show()
plt.plot(seasonal, label = 'Seasonal')
plt.legend(loc = 'upper right')

residual = decomposition.resid
plt.show()
plt.plot(residual, label = 'Residual')
plt.legend(loc='best')


# In[28]:


from statsmodels.tsa.stattools import adfuller
result = adfuller(energy['price actual'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))


# In[29]:


energy['total supply'] = energy['generation biomass']+energy['generation fossil gas']+energy['generation fossil brown coal/lignite']+energy['generation fossil hard coal']+energy['generation fossil oil']+energy['generation hydro pumped storage consumption']+energy['generation hydro run-of-river and poundage']+energy['generation hydro water reservoir']+energy['generation nuclear']+energy['generation other']+energy['generation other renewable']+energy['generation solar']+energy['generation waste']+energy['generation wind onshore']


# In[30]:


axes = energy['generation fossil gas'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='fossil gas')
axes = energy['generation fossil hard coal'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='fossil hard coal')
axes = energy['generation nuclear'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='nuclear')
axes = energy['generation wind onshore'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='wind onshore')


axes = energy['generation other'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='other')
axes = energy['generation other renewable'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='other renewable')
axes = energy['generation waste'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='waste')
axes = energy['generation hydro water reservoir'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='generation hydro water reservoir')
axes = energy['generation hydro pumped storage consumption'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='hydro pumped storage consumption')
axes = energy['generation hydro run-of-river and poundage'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='hydro run-of-river and poundage')
axes = energy['generation solar'].resample("M").mean().plot(marker='.', alpha=0.8,  figsize=(25,10), label='solar')

axes.legend(loc='upper right', frameon=False, fontsize=30)
axes.set_title('Generation Amount by Type', fontsize=30)
axes.set_ylabel('Monthly mean Generation Amount (GMh)', fontsize=20)
axes.set_xlabel("Year", fontsize=20)
axes.legend(loc=(1.01, .01), ncol=1, fontsize=15)
plt.tight_layout()


# In[31]:


# Calculate the percentages
print('Ordered percentage of total power generated over the 4 years by each source ')
print()
print("generation nuclear                         ",round((energy['generation nuclear'].sum()/energy['total supply'].sum())*100,1),'%')
print("generation fossil gas                      ",round((energy['generation fossil gas'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation wind onshore                    ',round((energy['generation wind onshore'].sum()/energy['total supply'].sum())*100,1),'%')
print("generation fossil hard coal                ",round((energy['generation fossil hard coal'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation hydro water reservoir            ',round((energy['generation hydro water reservoir'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation solar                            ',round((energy['generation solar'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation hydro run-of-river and poundage  ',round((energy['generation hydro run-of-river and poundage'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation hydro pumped storage consumption ',round((energy['generation hydro pumped storage consumption'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation waste                            ',round((energy['generation waste'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation other renewable                  ',round((energy['generation other renewable'].sum()/energy['total supply'].sum())*100,1),'%')
print('generation other                            ',round((energy['generation other'].sum()/energy['total supply'].sum())*100,1),'%')


# ## Deal with missing values

# In[32]:


energy = energy.fillna(method='ffill')


# In[33]:


energy.isnull().sum()


# ## Deal with outliers

# In[34]:


df1=energy.select_dtypes(exclude=['object'])


# In[35]:


import seaborn as sns


# In[36]:


plt.rcParams.update({'figure.max_open_warning': 0})


# In[37]:


for column in df1:
        plt.figure(figsize=(12,1))
        sns.boxplot(data=energy, x=column)


# In[38]:


Q1 = energy.quantile(0.25)
Q3 = energy.quantile(0.75)
IQR = Q3 - Q1
((energy < (Q1 - 1.5 * IQR)) | (energy > (Q3 + 1.5 * IQR))).sum()


# In[39]:


energy_out = energy[~((energy < (Q1 - 1.5 * IQR)) |(energy > (Q3 + 1.5 * IQR))).any(axis=1)]
print(energy_out.shape)


# In[40]:


print(energy['generation biomass'].quantile(0.50)) 
print(energy['generation biomass'].quantile(0.95)) 


# In[41]:


energy['generation biomass'] = np.where(energy['generation biomass'] > 543, 367, energy['generation biomass'])


# In[42]:


energy[0:-1] = np.where((energy[0:-1] < (Q1 - 1.5 * IQR)) | (energy[0:-1] > (Q3 + 1.5 * IQR)),energy[0:-1].quantile(0.50), energy[0:-1])


# In[43]:


energy.describe().T


# In[44]:


df2=weather.select_dtypes(exclude=['object'])


# In[45]:


for column in df2:
       plt.figure(figsize=(12,1))
       sns.boxplot(data=weather, x=column)


# In[46]:


weather.loc[weather.wind_speed > 50, 'wind_speed'] = np.nan


# In[47]:


weather.interpolate(method='linear', limit_direction='forward', inplace=True, axis=0)


# In[48]:


weather['time'] = pd.to_datetime(weather['dt_iso'], utc=True, infer_datetime_format=True)
weather = weather.drop(['dt_iso'], axis=1)
weather = weather.set_index('time')


# In[49]:


def df_convert_dtypes(df, convert_from, convert_to):
    cols = df.select_dtypes(include=[convert_from]).columns
    for col in cols:
        df[col] = df[col].values.astype(convert_to)
    return df


# In[50]:


weather = df_convert_dtypes(weather, np.int64, np.float64)


# In[51]:


print('There are {} missing values or NaNs in df_weather.'
      .format(weather.isnull().values.sum()))

temp_weather = weather.duplicated(keep='first').sum()

print('There are {} duplicate rows in df_weather based on all columns.'
      .format(temp_weather))


# In[52]:


weather_2 = weather.reset_index().drop_duplicates(subset=['time', 'city_name'], 
                                                        keep='last').set_index('time')

weather = weather.reset_index().drop_duplicates(subset=['time', 'city_name'],
                                                      keep='first').set_index('time')


# Reference: Kaggle (https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda)

# ### Handling categorical data

# In[53]:


uniqval_weather_des = weather['weather_description'].unique()
uniqval_weather_des


# In[54]:


uniqval_weather_main = weather['weather_main'].unique()
uniqval_weather_main


# In[55]:


uniqval_weather_icon = weather['weather_icon'].unique()
uniqval_weather_icon


# In[56]:


uniqval_weather_id = weather['weather_id'].unique()
uniqval_weather_id


# In[57]:


weather = weather.drop(['weather_main', 'weather_description', 'weather_icon'], axis=1)


# In[58]:


from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)


# In[59]:


ohe.fit(weather[['weather_id']])


# In[60]:


ohe.transform(weather[['weather_id']])


# ### Marge the two dataset

# In[61]:


df_1, df_2, df_3, df_4, df_5 = [x for _, x in weather.groupby('city_name')]
dfs = [df_1, df_2, df_3, df_4, df_5]


# In[62]:


# Merge all dataframes into the final dataframe

final_df = energy

for df in dfs:
    city = df['city_name'].unique()
    city_str = str(city).replace("'", "").replace('[', '').replace(']', '').replace(' ', '')
    df = df.add_suffix('_{}'.format(city_str))
    final_df = final_df.merge(df, on=['time'], how='outer')
    final_df = final_df.drop('city_name_{}'.format(city_str), axis=1)
    
final_df.columns


# In[63]:


final_df.describe().T


# Reference: kaggle https://www.kaggle.com/code/dimitriosroussis/electricity-price-forecasting-with-dnns-eda

# ### Correlation analysis

# In[64]:


final_df["wind load ratio"] = final_df["generation wind onshore"]/final_df["total load actual"]


# In[65]:


correlations = final_df.corr(method='pearson')
print(correlations['price actual'].sort_values(ascending=False).to_string())


# In[66]:


plt.figure(figsize=(25,12.10))
sns.heatmap(round(energy.corr(),1),annot=True,cmap='Blues',linewidth=0.9)
plt.show();


# ## Split data to train-test sets

# In[67]:


from sklearn.model_selection import train_test_split


# In[68]:


y = final_df['price actual']
X = final_df.drop('price actual', axis=1)


# In[69]:


X_train_full, X_test, y_train_full, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[70]:


# Split full training into training and val sets
# Make a 60-20-20 split
train_size = int(0.75 * X_train_full.shape[0])


# In[71]:


X_train, X_val, y_train, y_val = X_train_full[:train_size], X_train_full[train_size:], y_train_full[:train_size], y_train_full[train_size:]


# In[72]:


print(len(X_train),len(X_val),len(X_test))


# # Feature Engineering

# ## Feature selection

# In[73]:


from sklearn.feature_selection import VarianceThreshold


# In[74]:


var_thres = VarianceThreshold(threshold=0)
var_thres.fit(X_train)


# In[75]:


var_thres.get_support()


# In[76]:


constant_columns = [column for column in X_train.columns 
                   if column not in X_train.columns[var_thres.get_support()]]
print(len(constant_columns))


# In[77]:


X_train.drop(constant_columns,axis=1)


# ## Feature scaling
# Machine Learning algorithms don’t perform well when the input numerical attributes have very different scales

# In[78]:


from sklearn.preprocessing import StandardScaler


# In[79]:


scaler = StandardScaler()


# In[80]:


scaler.fit(X_train)


# In[81]:


scaled_data = scaler.transform(X_train)


# In[82]:


X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_val_scaled = scaler.transform(X_val)


# In[83]:


scaler.mean_


# In[84]:


scaler.scale_


# In[85]:


X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns)


# In[86]:


np.round(X_train_scaled.describe(),2)


# ## PCA

# In[87]:


from sklearn.decomposition import PCA


# In[88]:


pca = PCA(n_components=2)


# In[89]:


pca.fit(scaled_data)


# In[90]:


x_pca=pca.transform(scaled_data)


# In[91]:


x_pca.shape


# In[92]:


x_pca


# In[93]:


pca.explained_variance_ratio_


# In[94]:


plt.figure(figsize=(8,6))
plt.scatter(x_pca[:,0],x_pca[:,1],alpha=0.2)
plt.xlabel("First principle component")
plt.ylabel("Second principle component")


# # Modelling

# ## Multiple linear regression model

# In[95]:


## create X and y for multi linear regression
X = X_train[['generation fossil hard coal', 'total load actual','generation fossil gas',
             'generation fossil brown coal/lignite','wind load ratio']]
y = y_train
X_val = X_val[['generation fossil hard coal', 'total load actual','generation fossil gas',
             'generation fossil brown coal/lignite','wind load ratio']]


# In[96]:


from sklearn import linear_model
import statsmodels.api as sm


# In[97]:


lr_model = linear_model.LinearRegression()
lr_model.fit(X, y)

print('Intercept: \n', lr_model.intercept_)
print('Coefficients: \n', lr_model.coef_)


# In[98]:


prediction = lr_model.predict(X)
prediction


# In[99]:


lr_pred_X = lr_model.predict(X)
lr_pred_X
comparison_X_lr = pd.DataFrame({'Actual': y, 'Predicted': lr_pred_X})
print(comparison_X_lr)
comparison_X1_lr = comparison_X_lr.head(2000)
comparison_X1_lr.plot(kind="line", figsize=(10,8))
plt.title('Actual vs Predicted Values')
plt.xlabel("X: independant variables") 
plt.ylabel("y: Price Actual")
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.show()


# In[100]:


import sklearn.metrics as sm
from sklearn.metrics import mean_squared_error, mean_absolute_error, accuracy_score


# In[101]:


print("Mean absolute error =", round(sm.mean_absolute_error(y_train, lr_pred_X), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_train, lr_pred_X), 2)) 
mse_lr = mean_squared_error(y_train, lr_pred_X)
rmse_lr = mse_lr**.5
print("RMSE =", round(rmse_lr))


# ## Performance evaluation

# In[102]:


# prediction on the validation set
y_val_predict = lr_model.predict(X_val)
print("Mean absolute error =", round(sm.mean_absolute_error(y_val, y_val_predict), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_val, y_val_predict), 2)) 
mse_lr = mean_squared_error(y_val, y_val_predict)
rmse_lr = mse_lr**.5
print("RMSE =", round(rmse_lr))  


# In[170]:


from sklearn.model_selection import cross_val_score

#calculate 5-fold RMSE scores
linear_regression_scores = np.sqrt(-cross_val_score(lr_model,X,y, scoring="neg_mean_squared_error", cv=5))
# print mean and std of the scores
print("Linear regression: mean",linear_regression_scores.mean(), "sd:", linear_regression_scores.std())


# In[168]:





# ## Random forest model

# In[104]:


from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, max_features=60, max_leaf_nodes=None)                            
rf_model.fit(X_train,y_train);
rf_pred_X = rf_model.predict(X_train)


# In[105]:


comparison_X_rf = pd.DataFrame({'Actual': y_train, 'Predicted': rf_pred_X})
print(comparison_X_rf)
comparison_X_rf1 = comparison_X_rf.head(2000)
comparison_X_rf1.plot(kind="line", figsize=(10,8))

plt.title('Actual vs Predicted Values')
plt.xlabel("X: independant variables") 
plt.ylabel("y: Price Actual")
plt.grid()
plt.show()


# In[106]:


print("Mean absolute error =", round(sm.mean_absolute_error(y_train, rf_pred_X), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_train, rf_pred_X), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_train, rf_pred_X), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_train, rf_pred_X), 2)) 
print("R2 score =", round(sm.r2_score(y_train, rf_pred_X), 2))
mse_rf = mean_squared_error(y_train, rf_pred_X)
rmse_rf = mse_rf**.5
print("RMSE =", round(rmse_rf))


# ## Performance evaluation

# In[107]:


# random forest model on test data
rf_pred_X_test = rf_model.predict(X_test)

comparison_X_rf_test = pd.DataFrame({'Actual': y_test, 'Predicted': rf_pred_X_test})
print(comparison_X_rf_test)
comparison_X_rf1_test = comparison_X_rf_test.head(2000)
comparison_X_rf1_test.plot(kind="line", figsize=(10,8))

plt.title('Actual vs Predicted Values')
plt.xlabel("X: independant variables") 
plt.ylabel("y: Price Actual")
plt.grid()
plt.show()

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, rf_pred_X_test), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, rf_pred_X_test), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, rf_pred_X_test), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, rf_pred_X_test), 2)) 
print("R2 score =", round(sm.r2_score(y_test, rf_pred_X_test), 2))
mse_rf_test = mean_squared_error(y_test, rf_pred_X_test)
rmse_rf_test = mse_rf_test**.5
print("RMSE =", round(rmse_rf_test))


# ## XGBooter model

# In[108]:


# pip install xgboost


# In[109]:


import xgboost as xgb
xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=7, subsample=0.7, 
                             colsample_bytree = 0.3, learning_rate = 0.1, alpha = 10)
xgb_model.fit(X_train,y_train);
xgb_pred_X = xgb_model.predict(X_train)


# In[110]:


xgb_comparison_X = pd.DataFrame({'Actual': y_train, 'Predicted': xgb_pred_X})
print(xgb_comparison_X)
xgb_comparison_X1 = xgb_comparison_X.head(2000)
xgb_comparison_X1.plot(kind="line", figsize=(10,8))
plt.title('XGB Model - Actual vs Predicted Values')
plt.xlabel("X: independant variables") 
plt.ylabel("y: Price Actual")
plt.grid()
plt.show()


# In[111]:


print("Mean absolute error =", round(sm.mean_absolute_error(y_train, xgb_pred_X), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_train, xgb_pred_X), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_train, xgb_pred_X), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_train, xgb_pred_X), 2)) 
print("R2 score =", round(sm.r2_score(y_train, xgb_pred_X), 2))

mse_xgb = mean_squared_error(y_train, xgb_pred_X)
rmse_xgb = mse_xgb**.5
print("RMSE =", round(rmse_xgb))


# ## Performance evaluation

# In[112]:


# XGB model on test data
xgb_pred_X_test = xgb_model.predict(X_test)

comparison_X_xgb_test = pd.DataFrame({'Actual': y_test, 'Predicted': xgb_pred_X_test})
print(comparison_X_rf_test)
comparison_X_xgb1_test = comparison_X_xgb_test.head(2000)
comparison_X_xgb1_test.plot(kind="line", figsize=(10,8))

plt.title('Actual vs Predicted Values')
plt.xlabel("X: independant variables") 
plt.ylabel("y: Price Actual")
plt.grid()
plt.show()

print("Mean absolute error =", round(sm.mean_absolute_error(y_test, xgb_pred_X_test), 2)) 
print("Mean squared error =", round(sm.mean_squared_error(y_test, xgb_pred_X_test), 2)) 
print("Median absolute error =", round(sm.median_absolute_error(y_test, xgb_pred_X_test), 2)) 
print("Explain variance score =", round(sm.explained_variance_score(y_test, xgb_pred_X_test), 2)) 
print("R2 score =", round(sm.r2_score(y_test, xgb_pred_X_test), 2))
mse_xgb_test = mean_squared_error(y_test, xgb_pred_X_test)
rmse_xgb_test = mse_xgb_test**.5
print("RMSE =", round(rmse_xgb_test))


# ## Time Series Model

# In[113]:


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam


# In[114]:


timesteps = energy.index.to_numpy()
prices = energy["price actual"].to_numpy()

timesteps[:10], prices[:10]


# In[115]:


# Create train and test splits for time series data
split_size = int(0.8 * len(prices)) # 80% train, 20% test
X_train_prep, y_train_prep = timesteps[:split_size], prices[:split_size]
X_test_prep, y_test_prep = timesteps[split_size:], prices[split_size:]

len(X_train), len(X_test), len(y_train), len(y_test)


# In[116]:


plt.figure(figsize=(10, 7))
plt.scatter(X_train_prep, y_train_prep, s=5, label="Train data")
plt.scatter(X_test_prep, y_test_prep, s=5, label="Test data")
plt.xlabel("Date")
plt.ylabel("Electricity Price")
plt.legend(fontsize=14)
plt.show();


# In[117]:


# Create a naïve forecast
naive_forecast = y_test_prep[:-1] 
naive_forecast[:10], naive_forecast[-10:]


# In[118]:


# Create a function to plot time series data
def plot_time_series(timesteps, values, format='.', start=0, end=None, label=None):
  """
  Plots a timesteps (a series of points in time) against values (a series of values across timesteps).
  
  Parameters
  ---------
  timesteps : array of timesteps
  values : array of values across time
  format : style of plot, default "."
  start : where to start the plot (setting a value will index from start of timesteps & values)
  end : where to end the plot (setting a value will index from end of timesteps & values)
  label : label to show on plot of values
  """
  # Plot the series
  plt.plot(timesteps[start:end], values[start:end], format, label=label)
  plt.xlabel("Time")
  plt.ylabel("Electricity Price")
  if label:
    plt.legend(fontsize=14) 
  plt.grid(True)


# Reference: https://machinelearningmastery.com/time-series-data-visualization-with-python/

# In[119]:


# Plot naive forecast
plt.figure(figsize=(10, 7))
plot_time_series(timesteps=X_test_prep, values=y_test_prep, label="Test data")
plot_time_series(timesteps=X_test_prep[1:], values=naive_forecast, format="-",label="Naive forecast");


# In[120]:


import tensorflow as tf
from tensorflow.keras import layers


# In[121]:


def mean_absolute_scaled_error(y_true, y_pred):
  """
  Implement MASE (assuming no seasonality of data).
  """
  mae = tf.reduce_mean(tf.abs(y_true - y_pred))

  mae_naive_no_season = tf.reduce_mean(tf.abs(y_true[1:] - y_true[:-1])) # our seasonality is 1 day (hence the shifting of 1 day)

  return mae / mae_naive_no_season


# In[122]:


def evaluate_preds(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)
  
  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}


# In[123]:


naive_results = evaluate_preds(y_true=y_test_prep[1:],
                               y_pred=naive_forecast)
naive_results


# 

# In[124]:


def evaluate_preds_test(y_test, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  # Calculate various metrics
  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred) # puts and emphasis on outliers (all errors get squared)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)
  
  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}


# ## LSTM

# In[125]:


HORIZON = 24 # predict next 24 hours
WINDOW_SIZE = 168 # use last 1 week timesteps to predict the horizon


# In[126]:


# Create function to label windowed data
def get_labelled_windows(x, horizon=1):
  """
  Creates labels for windowed dataset.

  E.g. if horizon=1 (default)
  Input: [1, 2, 3, 4, 5, 6] -> Output: ([1, 2, 3, 4, 5], [6])
  """
  return x[:, :-horizon], x[:, -horizon:]


# In[127]:


# Create function to view NumPy arrays as windows 
def make_windows(x, window_size=24, horizon=24):
  """
  Turns a 1D array into a 2D array of sequential windows of window_size.
  """
  # 1. Create a window of specific window_size (add the horizon on the end for later labelling)
  window_step = np.expand_dims(np.arange(window_size+horizon), axis=0)
  # print(f"Window step:\n {window_step}")

  # 2. Create a 2D array of multiple window steps (minus 1 to account for 0 indexing)
  window_indexes = window_step + np.expand_dims(np.arange(len(x)-(window_size+horizon-1)), axis=0).T # create 2D array of windows of size window_size
  # print(f"Window indexes:\n {window_indexes[:3], window_indexes[-3:], window_indexes.shape}")

  # 3. Index on the target array (time series) with 2D array of multiple window steps
  windowed_array = x[window_indexes]

  # 4. Get the labelled windows
  windows, labels = get_labelled_windows(windowed_array, horizon=horizon)

  return windows, labels


# In[128]:


full_windows, full_labels = make_windows(prices, window_size=WINDOW_SIZE, horizon=HORIZON)
len(full_windows), len(full_labels)


# In[129]:


# View the first 3 windows/labels
for i in range(3):
  print(f"Window: {full_windows[i]} -> Label: {full_labels[i]}")


# In[130]:


# Make the train/test splits
def make_train_test_splits(windows, labels, test_split=0.2):
  """
  Splits matching pairs of windows and labels into train and test splits.
  """
  split_size = int(len(windows) * (1-test_split)) # this will default to 80% train/20% test
  train_windows = windows[:split_size]
  train_labels = labels[:split_size]
  test_windows = windows[split_size:]
  test_labels = labels[split_size:]
  return train_windows, test_windows, train_labels, test_labels


# In[131]:


train_windows, test_windows, train_labels, test_labels = make_train_test_splits(full_windows, full_labels)
len(train_windows), len(test_windows), len(train_labels), len(test_labels)


# In[132]:


train_windows[:5], train_labels[:5]


# In[133]:


tf.random.set_seed(42)
inputs = layers.Input(shape=(WINDOW_SIZE))
x = layers.Lambda(lambda x: tf.expand_dims(x, axis=1))(inputs)
x = layers.LSTM(128, activation="relu")(x) 
output = layers.Dense(HORIZON)(x)
LSTM_model = tf.keras.Model(inputs=inputs, outputs=output, name="model_lstm")

LSTM_model.summary()
# Compile model
LSTM_model.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam())

LSTM_model.fit(train_windows,
            train_labels,
            epochs=100,
            verbose=0,
            batch_size=128,
            validation_data=(test_windows, test_labels))


# In[134]:


test_windows[0]


# In[135]:


test_labels[0]


# In[136]:


LSTM_model.evaluate(test_windows, test_labels)


# In[137]:


def make_preds(model, input_data):
  """
  Uses model to make predictions on input_data.

  Parameters
  ----------
  model: trained model 
  input_data: windowed input data (same kind of data model was trained on)

  Returns model predictions on input_data.
  """
  forecast = model.predict(input_data)
  return tf.squeeze(forecast)


# In[138]:


def evaluate_preds(y_true, y_pred):
  y_true = tf.cast(y_true, dtype=tf.float32)
  y_pred = tf.cast(y_pred, dtype=tf.float32)

  mae = tf.keras.metrics.mean_absolute_error(y_true, y_pred)
  mse = tf.keras.metrics.mean_squared_error(y_true, y_pred)
  rmse = tf.sqrt(mse)
  mape = tf.keras.metrics.mean_absolute_percentage_error(y_true, y_pred)
  mase = mean_absolute_scaled_error(y_true, y_pred)

  if mae.ndim > 0: 
    mae = tf.reduce_mean(mae)
    mse = tf.reduce_mean(mse)
    rmse = tf.reduce_mean(rmse)
    mape = tf.reduce_mean(mape)
    mase = tf.reduce_mean(mase)

  return {"mae": mae.numpy(),
          "mse": mse.numpy(),
          "rmse": rmse.numpy(),
          "mape": mape.numpy(),
          "mase": mase.numpy()}


# In[139]:


# Make predictions with LSTM model
LSTM_model_preds = make_preds(LSTM_model, test_windows)
LSTM_model_preds[:10]


# In[140]:


# Evaluate LSTM_model preds
LSTM_model_results = evaluate_preds(y_true=tf.squeeze(test_labels),
                                 y_pred=LSTM_model_preds)
LSTM_model_results


# ### Multivariate_LSTM

# In[141]:


#pip install keras


# In[142]:


from keras.preprocessing.sequence import TimeseriesGenerator
from keras import Sequential
from keras.layers import SimpleRNN, Dense


# In[143]:


from sklearn.preprocessing import MinMaxScaler


# In[144]:


y1 = final_df['price actual']
X1 = final_df.drop('price actual', axis=1)


# In[145]:


scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X1)


# In[146]:


X_scaled


# In[147]:


len(X_scaled)


# In[148]:


sc = MinMaxScaler()
y = y1.values
y_scaled = sc.fit_transform(y.reshape(-1,1))


# In[149]:


y_scaled


# In[150]:


features = X_scaled
target = y_scaled


# In[151]:


TimeseriesGenerator(features, target, length=2, sampling_rate=1, batch_size=1)[0]


# In[152]:


x_train_prep,x_test_prep,y_train_prep,y_test_prep = train_test_split(features, target, test_size=0.2, random_state=123,shuffle = False)


# In[153]:


x_train_prep.shape


# In[154]:


x_test_prep.shape


# In[155]:


y_train_prep


# In[156]:


win_length=168
batch_size=32
num_features=65
train_generator = TimeseriesGenerator(x_train_prep,y_train_prep,length=win_length,sampling_rate=1,batch_size=batch_size)
test_generator = TimeseriesGenerator(x_test_prep,y_test_prep,length=win_length,sampling_rate=1,batch_size=batch_size)


# In[157]:


len(test_generator)


# In[158]:


mul_lstm = tf.keras.Sequential()
mul_lstm.add(tf.keras.layers.LSTM(128, input_shape= (win_length, num_features), return_sequences=True))
mul_lstm.add(tf.keras.layers.LeakyReLU(alpha=0.5))

mul_lstm.add(tf.keras.layers.LSTM(128,return_sequences=True))
mul_lstm.add(tf.keras.layers.LeakyReLU(alpha=0.5))
mul_lstm.add(tf.keras.layers.Dropout(0.3))

mul_lstm.add(tf.keras.layers.LSTM(64,return_sequences=False))
mul_lstm.add(tf.keras.layers.Dropout(0.3))
mul_lstm.add(tf.keras.layers.Dense(1))


# In[159]:


mul_lstm.summary()


# In[160]:


early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                    mode='min')


# In[161]:


len(test_generator)


# In[162]:


mul_lstm.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(),
                metrics=[tf.metrics.RootMeanSquaredError()])



history = mul_lstm.fit_generator(train_generator,
                                 epochs=50,
                                 validation_data = test_generator,
                                 shuffle=False,
                                 callbacks=[early_stopping])


# In[163]:


predictions = mul_lstm.predict_generator(test_generator) 


# In[164]:


len (predictions)


# In[165]:


df_pred=pd.concat([pd.DataFrame(predictions),pd.DataFrame(y_test)],axis=1)


# ## Hyperparameter tuning

# In[171]:


from sklearn.model_selection import RandomizedSearchCV

distributions = dict(n_estimators=range(1,200), max_features=(1,65), max_leaf_nodes=(10,100))      

# apply the randomised search
random_search = RandomizedSearchCV(rf_model, distributions, cv=10, 
                           scoring="neg_mean_squared_error", 
                           return_train_score=True, random_state=42, n_jobs=-1)

# fit to data
random_search.fit(X_train,y_train)


# In[177]:


#check the best hyperparameter values
random_search.best_estimator_

# assign the best model to final_model object
final_model = random_search.best_estimator_


# In[180]:


final_predictions = final_model.predict(X_test)
final_mse = mean_squared_error(y_test,final_predictions)
final_rmse = np.sqrt(final_mse)

