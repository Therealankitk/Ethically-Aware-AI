from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

data = pd.read_csv('Datasets/compas.csv')

prot_attr = ['sex','age','race']
targets = ['two_year_recid']

X = data.drop(['two_year_recid'], axis = 1)
y = data['two_year_recid']

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

#model

alch = RandomForestClassifier(n_estimators=100, random_state=42)

#train

alch.fit(X_train,y_train)

#model preservation

filename = 'models/alchModel1.sav'
joblib.dump(alch,filename)

print("Training complete!")








