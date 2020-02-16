# import libraries
import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model, preprocessing
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier

# loading data
data = pd.read_csv('KNN/car.data')
print(data.head())  # display first 5 rows

# converting non numerical data to integers
le = preprocessing.LabelEncoder()
buying = le.fit_transform(list(data['buying']))
maint = le.fit_transform(list(data['maint']))
door = le.fit_transform(list(data['door']))
persons = le.fit_transform(list(data['persons']))
lug_boot = le.fit_transform(list(data['lug_boot']))
safety = le.fit_transform(list(data['safety']))
cls = le.fit_transform(list(data['class']))

# recombining our data into a features list and label list
x = list(zip(buying, maint, door, persons, lug_boot, safety))  # features
y = list(cls)  # labels

# splitting data into training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# creating the knn model
model = KNeighborsClassifier(n_neighbors=9)

# training our model
model.fit(x_train, y_train)

# calculating the models accuracy
acc = model.score(x_test, y_test)
print('Accuracy: ', acc)

# testing our model
predicted = model.predict(x_test)
names = ['unacc', 'acc', 'good', 'vgood']

for x in range(len(predicted)):
    print('Predicted: ', names[predicted[x]], 'Data: ', x_test[x], 'Actual: ', names[y_test[x]])
    # will display predicted class, our data, and the actual class
    # we create a names list so that we can convert our integer predictions into their string representations

    # looking at neighbours
    # we will see the neighbours at each point in our testing data
    n = model.kneighbors([x_test[x]], 9, True)
    print('N: ', n)
