# Importing the libraries
import numpy as np
import pandas as pd
import tensorflow as tf
tf.__version__


# Importing the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values
y = dataset.iloc[:, -1].values

# Encoding categorical data
# Label Encoding the "Gender" column
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X_1 = LabelEncoder()
labelencoder_X_2 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
#for creating dummy variables
#not working
# onehotencoder = OneHotEncoder(categorical_features = [1]
# X = onehotencoder.fit_transform(X).toarray()
#working
# One Hot Encoding the "Geography" column
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


#Import keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense


# # Initializing the ANN
# classifier = Sequential()

# # Adding the input layer and the first hidden layer
# #chosen 6 nodes in the hidden layer because it is average of (no. of features or columns in training set or our input layer) and (no. of nodes or columns in output layer), hence (11+2)/2 = 6
# classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim = 11))

# # Adding the second hidden layer
# classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))

# # Adding the output layer
# classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))

# # Compiling the ANN
# classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# # Training the ANN on the Training set
# classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
 


y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)




#Using K-Fold Cross Validation

#Evaluating the ANN
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score

#function for building the architecture of our ANN classifier
def build_classifier():
    #Building the architecture  of the ANN classifier
    classifier = Sequential()
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu', input_dim = 11))
    classifier.add(Dense(units=6, kernel_initializer='uniform', activation='relu'))
    classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return classifier

classifier = KerasClassifier(build_fn = build_classifier, batch_size = 10, epochs = 100)
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10, n_jobs = -1)
mean = accuracies.mean()
variance = accuracies.std()
print(mean)
print(variance)


    





