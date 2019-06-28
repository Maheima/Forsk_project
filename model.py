import pandas  as pd
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
import pickle
#import requests
#import json

data = pd.read_csv("phpjX67St.csv")
data.head(3)
data.columns = ["Diagnosis","Forced vital capacity","Volume that has been exhaled","Performance status",
                "Pain before surgery","Haemoptysis before surgery","Dyspnoea before surgery","Cough before surgery",
                "Weakness before surgery","T in clinical TNM","diabetes mellitus","MI up to 6 months",
                "peripheral arterial diseases","Smoking","Asthma","Age at surgery","1 year survival"]
pd.value_counts(data['1 year survival']).plot.bar()
plt.title('Fraud class histogram')
plt.xlabel('1 year survival')
plt.ylabel('Frequency')
data['1 year survival'].value_counts()


X = np.array(data.iloc[:,:16])
y = np.array(data.iloc[:,16])
# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

#print('Shape of X: {}'.format(X.shape))
#print('Shape of y: {}'.format(y.shape))
#
#
#print("Number transactions X_train dataset: ", X_train.shape)
#print("Number transactions y_train dataset: ", y_train.shape)
#print("Number transactions X_test dataset: ", X_test.shape)
#print("Number transactions y_test dataset: ", y_test.shape)
#
#print("Before OverSampling, counts of label '1': {}".format(sum(y_train==1)))
#print("Before OverSampling, counts of label '2': {} \n".format(sum(y_train==2)))

sm = SMOTE(random_state=2)
X_train_res, y_train_res = sm.fit_sample(X_train, y_train)
#
#print('After OverSampling, the shape of train_X: {}'.format(X_train_res.shape))
#print('After OverSampling, the shape of train_y: {} \n'.format(y_train_res.shape))
#
#print("After OverSampling, counts of label '1': {}".format(sum(y_train_res==1)))
#print("After OverSampling, counts of label '2': {}".format(sum(y_train_res==2)))
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
lr = LogisticRegression()
lr.fit(X_train_res,y_train_res)
probability = lr.predict_proba(X_test)
labels_pred = lr.predict(X_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, labels_pred)

pickle.dump(lr, open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))
print(model.predict([[3,2.44,0.96,3,1,2,1,2,2,1,1,1,1,2,1,73]]))