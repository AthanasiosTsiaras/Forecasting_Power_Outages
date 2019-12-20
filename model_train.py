# Loading the model data and usingi the model_ranking function to compare the
# performance of different models and also export the train models.

import pandas as pd
import numpy as np

df = pd.read_csv("model_dataset.csv")
df = pd.get_dummies(df, drop_first = True)

X = np.array(df.loc[:, df.columns != 'Y'])
y = np.array(df.Y)

from model_ranking import model_ranking

results, logit, dec_tree, rand_for, grad_bst = model_ranking(X, y)
results.to_csv('trained_models/results_train3.csv', index = False)

import pickle
models = ['trained_models/logit3.sav', 'trained_models/dec_tree3.sav',
'trained_models/rand_for3.sav', 'trained_models/grad_bst3.sav']
pickle.dump(logit, open(models[0], 'wb'))
pickle.dump(dec_tree, open(models[1], 'wb'))
pickle.dump(rand_for, open(models[2], 'wb'))
pickle.dump(grad_bst, open(models[3], 'wb'))

print(results)
