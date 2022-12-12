from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
data = pd.read_csv(r'C:\Users\stanley\Desktop\DataMining\miami.csv')
data.drop('age', axis=1, inplace=True)
data.drop('avno60plus', axis=1, inplace=True)
data.drop('month_sold', axis=1, inplace=True)
data.drop('PARCELNO', axis=1, inplace=True)
data.drop('LATITUDE', axis=1, inplace=True)
data.drop('LONGITUDE', axis=1, inplace=True)
data.loc[data['SALE_PRC'] < 399999, 'SALE_PRC'] = 0
data.loc[data['SALE_PRC'] > 399999, 'SALE_PRC'] = 1



X = data.iloc[:,[1,2,3,4,5,6,7,8,9,10]].values
y = data['SALE_PRC'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)
SVC_model = svm.SVC()
KNN_model = KNeighborsClassifier(n_neighbors=5)
SVC_model.fit(X_train, y_train)
KNN_model.fit(X_train, y_train)
SVC_prediction = SVC_model.predict(X_test)
KNN_prediction = KNN_model.predict(X_test)
print(accuracy_score(SVC_prediction, y_test))
print(accuracy_score(KNN_prediction, y_test))
print(confusion_matrix(SVC_prediction, y_test))
print(classification_report(KNN_prediction, y_test))


