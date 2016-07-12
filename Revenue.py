#!/usr/bin/python

import numpy as np
import pandas as pd
from pandas import DataFrame,Series,DatetimeIndex
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVR
from sklearn.linear_model import RANSACRegressor
from sklearn.linear_model import ElasticNet

from sklearn.ensemble import GradientBoostingRegressor

from math import sqrt
import time
from datetime import date,datetime

def modify_df(df):
	df.set_index(['Id'],drop=True,inplace=True)
	df.rename(columns={c:c.replace(" ","") for c in df.columns.tolist() if " " in c}, inplace=True)
	df['year'] = DatetimeIndex(df['OpenDate']).year
	df['month'] = DatetimeIndex(df['OpenDate']).month
	#df['day'] = DatetimeIndex(df['OpenDate']).day
	df['DaysOpened']=df['OpenDate']
	df['DaysOpened'] = df['DaysOpened'].map(lambda x: (date.today() - datetime.strptime(x,'%m/%d/%Y').date()).days)
	#df['dayofweek'] = DatetimeIndex(df['OpenDate']).dayofweek
	#df['CityNameLength']=df['City'].map(lambda x: len(x))
	df['CityGrp'] = df['CityGroup'].map( {'Big Cities': 0, 'Other': 1} )
	#df['TypeCode'] = df['Type'].map( {'IL': 0, 'FC': 1,'DT':2,'MB':3} )
	df.drop(['OpenDate','City','CityGroup','Type'],axis=1, inplace=True)
	'''
	col_dic={}
	for i in range(1,38):
	    col = "P{0}".format(i)
	    #print col, " - ",sorted(df[col].unique())
	    for val in sorted(df[col].unique()):
	    	new_col = col + "_{0}".format(val)
	    	col_dic[new_col] = val
	#print col_dic
	for key,val in col_dic.items():
		df[key] = 0
		col_exist = key.split("_")[0]
		df.ix[df[col_exist] == val, key] = 1
	'''
	return df

def create_submission_file(df_sub,pred):
	cols_remove = df_sub.columns.tolist()
	df_sub.reset_index(level=0,inplace=True)
	df_sub['Prediction']=pred
	df_sub.drop(cols_remove,axis=1, inplace=True)
	df_sub.to_csv('submission.csv',index=False)
	return


def rmse(y, predy):
	return sqrt(np.mean((y - predy)**2))

def RForest(x,y):
	print "Starting fit() with RandomForestRegressor"
	
	#Grid GridSearch
	params = {#'max_depth':[None,3],
			'max_depth':[3],
			#'n_estimators':[750, 1000, 1250],
			'n_estimators':[1250],
			#'min_samples_split':[1, 2], 
			'min_samples_split':[2],
			#'random_state':[None,42]
			}
	rf = RandomForestRegressor()
	clf = GridSearchCV(rf, params,cv=3)
	
	#clf = RandomForestRegressor(n_estimators=750, min_samples_split=1)
	start = time.time()
	clf = clf.fit(x,y)
	end = time.time()
	print "Fit completed in",time.strftime('%H:%M:%S',time.gmtime(end-start))
	return clf

def SVRegressor(x, y):
	print "Starting fit() with SVR"
	
	#Grid GridSearch
	params = {'C':[.001, .01, .1, 1e0, 1e1, 1e2], "gamma": np.logspace(-4, 0, 5),
				#'random_state':[None,42]
			}
	svr = SVR()
	clf = GridSearchCV(svr, params,cv=3)
	
	start = time.time()
	clf = clf.fit(x,y)
	end = time.time()
	print "Fit completed in",time.strftime('%H:%M:%S',time.gmtime(end-start))
	return clf

def RANSAC(x, y):
	print "Starting fit() with RANSAC Regressor"
	#Grid GridSearch
	clf = RANSACRegressor()
	start = time.time()
	clf = clf.fit(x,y)
	end = time.time()
	print "Fit completed in",time.strftime('%H:%M:%S',time.gmtime(end-start))
	return clf

def ENet(x, y):
	print "Starting fit() with ElasticNet"
	#Grid GridSearch
	params = {'alpha':np.linspace(0.1,1,5),
			'l1_ratio': np.linspace(0.1,.9,5),
			}
	en = ElasticNet(fit_intercept=True, normalize=True, warm_start=True)
	clf = GridSearchCV(en, params,cv=3)
	start = time.time()
	clf = clf.fit(x,y)
	end = time.time()
	print "Fit completed in",time.strftime('%H:%M:%S',time.gmtime(end-start))
	return clf

def GradientBoost(x, y):
	print "Starting fit() with Gradient Boosting Regressor"
	#Grid GridSearch
	
	params = {#'loss':['ls', 'huber'],
			'loss':['huber'],
			#'learning_rate': [0.05, 0.01],
			'learning_rate': [0.001],
			#'n_estimators':[1250,1500],
			'n_estimators':[1250],
			#'max_depth':[2,3],
			'max_depth':[2],
			#'min_samples_split':[1,2],
			'min_samples_split':[1],
			#'subsample':[.5]
			#'alpha':[.5,.9]
				#'random_state':[None,42]
			}
	gbr = GradientBoostingRegressor(warm_start=True)
	clf = GridSearchCV(gbr, params,cv=3)
	
	#clf = GradientBoostingRegressor(alpha= 0.9, learning_rate= 0.001, loss= 'huber', max_depth= 3, min_samples_split= 1, n_estimators= 1500)
	start = time.time()
	clf = clf.fit(x,y)
	end = time.time()
	print "Fit completed in",time.strftime('%H:%M:%S',time.gmtime(end-start))
	return clf

df = pd.read_csv('train.csv')
# Remove outliers - revenue > 10000000
df.drop(df.index[[16,75,99]],inplace=True)
df = modify_df(df)

print "Train dataset modified..."

df['log-revenue'] = df['revenue'].apply(lambda x: np.log(1 + x))
y = df['revenue']
y_log = df['log-revenue']
#df.drop(['revenue'],axis=1, inplace=True)
df.drop(['revenue', 'log-revenue'],axis=1, inplace=True)

#imp_cols = ['DaysOpened','P28','P20','month','P1','P17','P22','P2','P11','P6','P21','P3','year','P23','P19','P5','P4','P8','P25','P29','CityGrp']
#imp_cols = ['DaysOpened', 'P14_4', 'P20', 'P28_3.0', 'P28', 'P17_5', 'month', 'P1', 'P1_4', 'P20_2', 'P22', 'P19_15', 'P5_3', 'P2', 'P28_1.0', 'P31_5', 'year', 'P22_3', 'CityGrp', 'P21_3', 'P23_2', 'P5_2']
# Important columns for Gradient Boosting Regressor
#imp_cols = ['DaysOpened', 'P14_4', 'P28', 'month', 'P28_3.0', 'P31_5', 'P22_3', 'P20', 'P1', 'P23_2', 'P28_1.0', 'P5_3', 'P17_5', 'year', 'P21_3', 'P1_4', 'P2', 'P1_3', 'P22', 'P19_4', 'P17_1', 'P11_5', 'P19', 'P25_2', 'P30', 'CityGrp', 'P20_2', 'P23', 'P21', 'P21_2']
#df = df [imp_cols]

#df_train, df_test, y_train, y_test = train_test_split(df, y, test_size=0.3, random_state=42)
df_train, df_test, y_train, y_test = train_test_split(df, y_log, test_size=0.3, random_state=42)

# Use Random Forest Regressor
#clf = RForest(df_train, y_train)

#Train on the entire train dataset
#clf = RForest(df.values, y.values)

# Use SVR
#clf = SVRegressor(df_train, y_train)

# Use SVR
#clf = ENet(df_train, y_train)

# Use RANSAC Regressor
#clf = RANSAC(df_train, y_train)

# Use Gradient Boosting REgressor
#clf = GradientBoost(df_train, y_train)

#Train on the entire dataset
#clf = GradientBoost(df.values, y.values)
clf = GradientBoost(df.values, y_log.values)

#pred = clf.predict(df_test)
#print rmse(y_test, pred)


# Read test.csv file
df_sub = pd.read_csv('test.csv')
df_sub = modify_df(df_sub)
#df_sub = df_sub [imp_cols]

# Make predictions on the test dataset
pred_sub = clf.predict(df_sub)

# Create a submission file
#create_submission_file(df_sub, pred_sub)
create_submission_file(df_sub, np.exp(pred_sub)-1)
