import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense
import matplotlib.pyplot as plt

# create dataframe
df = pd.read_csv('cars.csv')
print(df.head())

# features
x = np.array(df[['age', 'gender', 'miles', 'debt', 'income']])
# target
y = np.array(df['sales'])

# split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=42)

# reshape labels array
y_train = np.reshape(y_train, (-1, 1))

# define the model
model = Sequential()
model.add(Dense(32, input_dim=5, kernel_initializer='normal', activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))

# compile the model
model.compile(loss='mse', optimizer='adam', metrics=['mse', 'mae'])

# fit the model
model.fit(x_train, y_train, epochs=200, batch_size=10)

# make prediction
y_pred = model.predict(x_test)

# plot real values against predicted values
plt.scatter(y_test, y_pred)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
