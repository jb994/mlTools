import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

class mler():
	def __init__(modelNames):
		self.modelNames=modelNames #list of strings corresponding to models

	def fit(X_train, y_train,
		paramsList={},
		):
		models={}
		for model in modelNames:
			func=eval(model)
			if model not in paramsList.keys():
				params[model]=None
			func = eval(model).fit(X_train, y_train, paramsList[model])
			models[model] = func
		self.models = models

	def score(X_test, y_test):
		for model in self.models:
			print(f"{model.__name__}: {model.score(X_test, y_test)}")


	def LinearRegression(X, y):
		from sklearn.linear_model import LinearRegression as model
		return model().fit(X, y)
	def Ridge(X, y):
		from sklearn.linear_model import Ridge as model
		return model().fit(X,y)
	def xgboost(X, y):
		import xgboost as model
		D_train = model.DMatrix(X_train, label=y_train)
		return model().fit(X,y)


def linearRegresser(X_test, y_test, X_train, y_train, params={}):
	from sklearn.linear_model import LinearRegression
	print("Linear Regression")
	model = LinearRegression().fit(X_train, y_train)
	print(model.score(X_test, y_test))
	return model

def ridgeRegresser(X_test, y_test, X_train, y_train, params={}):
	from sklearn.linear_model import Ridge
	print("Ridge Regression")
	model = LinearRegression().fit(X_train, y_train)
	print(model.score(X_test, y_test))
	return model

def xgbooster(X_test, y_test, X_train, y_train, 
	params={
        'eta':0.3, #Like lr; weights the additinos of newly added trees
        'max_depth':3, #Depth of decision trees to be trained
        #'objective':'multi:softprob', #Loss function
        'objective':'reg:tweedie', #Loss function
        #'num_class':3 #Num classes in dataset
    	}
    ):
	import xgboost as xgb
	D_train = xgb.DMatrix(X_train, label=y_train)
	D_test = xgb.DMatrix(X_test, label=y_test)

	steps=20
	model = xgb.XGBRegressor(**params).fit(X_train, y_train)
	print(model.score(X_test, y_test))
	return model



