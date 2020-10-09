import pandas as pd
from sklearn import metrics
import numpy as np
a = np.array([1,2,3])

dataset = pd.read_csv(r'C:\Users\Dell\Desktop\Train_and_test2.csv')
dataset.head()
dataset.columns

dataset = dataset.drop(['zero','zero.1','zero.2','zero.3','zero.4','zero.5','zero.6','zero.7','zero.8','zero.9','zero.10','zero.11','zero.12','zero.13','zero.14','zero.15','zero.16','zero.17','zero.18',],axis=1)
dataset.dropna(inplace=True)
dataset.rename( columns={'2urvived':'Survived'}, inplace=True)

y = dataset["Survived"]
X = dataset.drop('Survived', axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=53)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train , y_train)
pred = model.predict(X_test)

print('   score:', metrics.accuracy_score(pred, y_test))


from sklearn.metrics import classification_report
print(classification_report(y_test, pred))

