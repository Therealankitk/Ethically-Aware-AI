import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.ensemble import RandomForestClassifier
from fairlearn.reductions import ExponentiatedGradient, DemographicParity

data = pd.read_csv('Datasets/compas.csv')

# prot_attr = ['race']
# prot_attr = ['sex']
prot_attr = ['sex','race']

targets = ['two_year_recid']

X = data.drop(['two_year_recid'], axis = 1)
y = data['two_year_recid']
A = X[prot_attr]  

X_train,X_test,y_train,y_test,A_train,A_test = train_test_split(X,y,A,test_size=0.2,random_state=42)

method = DemographicParity()

clf = RandomForestClassifier(random_state=0)

alch2 = ExponentiatedGradient(estimator=clf, constraints=method)

alch2.fit(X_train, y_train, sensitive_features=A_train)

# Step 5: Predict on test data
fair_predictions = alch2.predict(X_test)

# Step 6: Evaluate accuracy
print("Fair model accuracy:", accuracy_score(y_test, fair_predictions))

filename = 'models/FairAlchModel.sav'
joblib.dump(alch2,filename)

print("Training done")


