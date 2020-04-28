import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.externals import joblib


#%% get dataset
def fetch_dataset():
	dataset_url = 'http://mlr.cs.umass.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
	data = pd.read_csv(dataset_url, sep=';')
	data.to_csv('wine.csv', index=False)


# fetch_dataset()

#%% load data
data = pd.read_csv('wine.csv')
print(data.head())
print(data.shape)

#%% split data into train and test set
y = data.quality
x = data.drop('quality', axis=1)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=123, stratify=y)

#%% declare data processing steps
pipeline = make_pipeline(preprocessing.StandardScaler(), RandomForestRegressor(n_estimators=100))

# declare hyperparameters to tune
hyperparameters = {'randomforestregressor__max_features': ['auto', 'sqrt', 'log2'],
                   'randomforestregressor__max_depth': [None, 5, 3, 1]}

# tune model using cross-validation pipeline
clf = GridSearchCV(pipeline, hyperparameters, cv=10)
clf.fit(x_train, y_train)

#%% evaluate model pipeline on test data
pred = clf.predict(x_test)
print('r2 score:', r2_score(y_test, pred))
print('MSE:', mean_squared_error(y_test, pred))

#%% save model for future use
joblib.dump(clf, 'rf_regressor.pkl')
