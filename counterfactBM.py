import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
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

data_dice = dice_ml.Data(
    dataframe=pd.concat([X_train, y_train], axis=1),
    continuous_features=[col for col in X_train.columns if col not in ['sex','age','race', 'two_year_recid']],
    categorical_features=['sex','age','race'],  
    outcome_name='two_year_recid'
)
dice_model = dice_ml.Model(model=alch,backend='sklearn')

explainations = Dice(data_dice,dice_model)

protected_attr = 'race'
prot_attr_2 = 'sex'
protected_values = [0, 1]  #  0 for african, 1 for caucasian

num_instances = 500
subset = X_test.sample(n=num_instances, random_state=42).copy()
subset_original = subset.copy()


subset_flipped = subset.copy()
subset_flipped[protected_attr] = 0
subset_flipped[prot_attr_2] = 1


y_pred_original = alch.predict(subset_original)
y_pred_flipped = alch.predict(subset_flipped)

print(accuracy_score(y_pred_original,y_pred_flipped))


#comparison
comparison_df = subset_original.copy()
comparison_df['protected_attr_original'] = subset_original[protected_attr]
comparison_df['protected_attr_flipped'] = subset_flipped[protected_attr]
comparison_df['prediction_original'] = y_pred_original
comparison_df['prediction_flipped'] = y_pred_flipped
comparison_df['changed_prediction'] = comparison_df['prediction_original'] != comparison_df['prediction_flipped']


print(comparison_df[[protected_attr, 'protected_attr_flipped', 'prediction_original', 'prediction_flipped', 'changed_prediction']])




plt.figure(figsize=(14, 8))
for i in range(len(y_pred_original)):
    plt.plot([i, i], [y_pred_original[i], y_pred_flipped[i]], 'k--', alpha=0.6)
    plt.plot(i, y_pred_original[i], 'bo', label='Original' if i == 0 else "")
    plt.plot(i, y_pred_flipped[i], 'ro', label='Modified' if i == 0 else "")

plt.xticks(np.arange(0,len(y_pred_original),50), [f'#{i}' for i in range(0,len(y_pred_original),50)])
plt.ylabel('Prediction')
plt.title('Prediction Change per Instance for biased model(Race Set to African-American(0) and Sex set to Male(1))')
plt.legend(fontsize=18)
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()