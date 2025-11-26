import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate
import dice_ml
from dice_ml import Dice

data = pd.read_csv('Datasets/compas.csv')

# prot_attr = ['race']
# prot_attr = ['sex']
prot_attr = ['sex','race']

targets = ['two_year_recid']

X = data.drop(['two_year_recid'], axis = 1)
y = data['two_year_recid']
A = X[prot_attr]  

X_train,X_test,y_train,y_test,A_train,A_test = train_test_split(X,y,A,test_size=0.2,random_state=42)

#jamming up the model
filename = 'models/alchModel1.sav'
alch = joblib.load(filename)

predictions = alch.predict(X_test)
accuracy = accuracy_score(predictions,y_test)

print("The accuracy of the model is:",accuracy)

y_pred = alch.predict(X_test)


metric_frame = MetricFrame(
    metrics={
        'accuracy': accuracy_score,
        'selection_rate': selection_rate
    },
    y_true=y_test,
    y_pred=y_pred,
    sensitive_features=A_test
)

print("\n📋 Fairness Metrics by Group:\n")
formatted_df = metric_frame.by_group.copy()
formatted_df.index.name = 'Group (Sex, Race)'
print(formatted_df.round(4).to_string())

metric_frame.by_group.plot(
    kind='bar',
    figsize=(10, 6),
    rot=45,
    title='Fairness Metrics by Group (Race and Sex)',
)

plt.ylabel("Metric Value")
plt.xlabel("Protected Group (Race and Sex)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Metric",fontsize=18)
plt.tight_layout()
plt.show()

# cm = confusion_matrix(predictions,y_test)
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.xlabel('Predictions')
# plt.ylabel('test set')
# plt.show()


