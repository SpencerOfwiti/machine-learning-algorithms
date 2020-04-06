# import libraries
import sklearn
from sklearn import svm, datasets, metrics
from sklearn.neighbors import KNeighborsClassifier

# load breast cancer dataset
cancer = datasets.load_breast_cancer()
print('Features: ', cancer.feature_names)
print('Labels: ', cancer.target_names)

# splitting data into training and testing data
x = cancer.data  # all of the features
y = cancer.target  # all of the labels

# splitting our data into training and testing data
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.2)

# create a model
clf = svm.SVC(kernel='linear', C=2)  # adding a kernel and setting the margin
clf.fit(x_train, y_train)

# predict values for our test data
y_pred = clf.predict(x_test)

# test our predictions against the correct data
acc = metrics.accuracy_score(y_test, y_pred)
print('SVM: ', acc)

# compare with KNN
knn = KNeighborsClassifier(n_neighbors=11)
knn.fit(x_train, y_train)
pred = knn.predict(x_test)
acc2 = metrics.accuracy_score(y_test, pred)
print('KNN: ', acc2)
