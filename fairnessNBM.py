import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
from fairlearn.metrics import MetricFrame, selection_rate

data = pd.read_csv('Datasets/compas.csv')

# prot_attr = ['race']
# prot_attr = ['sex']
prot_attr = ['sex','race']

targets = ['two_year_recid']

X = data.drop(['two_year_recid'], axis = 1)
y = data['two_year_recid']
A = X[prot_attr]  

X_train,X_test,y_train,y_test,A_train,A_test = train_test_split(X,y,A,test_size=0.2,random_state=42)
alch2 = joblib.load('models/FairAlchModel.sav')

fair_predictions = alch2.predict(X_test)
metric_frame_fair = MetricFrame(
    metrics={'accuracy': accuracy_score, 'selection_rate': selection_rate},
    y_true=y_test,
    y_pred=fair_predictions,
    sensitive_features=A_test
)

print("\nFair model performance by group:")
print(metric_frame_fair.by_group)


metric_frame_fair.by_group.plot(
    kind='bar',
    figsize=(8, 6),
    rot=0,
    title='Fairness Metrics by Group (Race and Sex)',
)

plt.ylabel("Metric Value")
plt.xlabel("Protected Group (Race and Sex)")
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.legend(title="Metric",fontsize=18)
plt.tight_layout()
plt.show()