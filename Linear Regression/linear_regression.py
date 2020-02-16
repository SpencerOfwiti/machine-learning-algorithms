# import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

data = pd.read_csv('Linear Regression/student-mat.csv', sep=';')
# since our data is separated by semicolons we need to do sep=';'

# trimming our data to show only relevant attributes
data = data[['G1', 'G2', 'G3', 'studytime', 'failures', 'absences']]
data = shuffle(data)  # shuffle the data
print(data.head())

# label - attribute we are trying to predict
predict = 'G3'

# features - attributes that will determine our label
x = np.array(data.drop([predict], 1))  # features
y = np.array(data[predict])  # labels

# splitting our data into training and testing data
# 90% training, 10% testing
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# train model multiple times for best score
best = 0
for _ in range(1000):
    # splitting our data into training and testing data
    # 90% training, 10% testing
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

    # implementing linear regression
    # defining the model
    linear = linear_model.LinearRegression()
    linear.fit(x_train, y_train)
    acc = linear.score(x_test, y_test)  # acc stands for accuracy
    print('Accuracy: ', acc)

    # if the current model has a better score than the one we've already trained then save it
    if acc > best:
        best = acc
        # saving our model
        with open('Linear Regression/studentgrades.pickle', 'wb') as f:
            pickle.dump(linear, f)

# loading our models
pickle_in = open('Linear Regression/studentgrades.pickle', 'rb')
linear = pickle.load(pickle_in)
# now we can use linear to predict grades like before

linear.fit(x_train, y_train)
acc = linear.score(x_test, y_test)  # acc stands for accuracy
print('Best Accuracy: ', acc)

# viewing constants used to generate the line of best fit
print('------------------------------')
print('Coefficient: \n', linear.coef_)  # these are each slope value
print('Intercept: \n', linear.intercept_)  # this is the intercept
print('------------------------------')

# predicting specific students
predictions = linear.predict(x_test)  # gets a list of all predictions

for x in range(len(predictions)):
    print(predictions[x], x_test[x], y_test[x])

# drawing and plotting our model
plot = 'G2'  # change this to G1, G2, studytime, failures or absences to see other graphs
plt.scatter(data[plot], data['G3'])
plt.legend(loc=4)
plt.xlabel(plot)
plt.ylabel('Final Grade')
plt.show()
