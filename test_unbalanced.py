# Loading the trained models and testing them on the unbalanced dataset to see
# which ones were able to scale

import pickle
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import classification_report
import random
random.seed(69)

test = pd.read_csv("2018_unbalanced.csv")
test = test.loc[:, test.columns !='Date']
test = pd.get_dummies(test, drop_first = True)
train = pd.read_csv("model_dataset.csv")
train = pd.get_dummies(train, drop_first = True)

list1 = list(train.columns)
list2 = list(test.columns)
missing_from_test = list(set(list1).difference(list2))
for c in missing_from_test:
    test[c] = 0

X = np.array(test.loc[:, test.columns != 'Y'])
y = np.array(test.Y)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 69)

models = ['trained_models/logit3.sav', 'trained_models/dec_tree3.sav',
          'trained_models/rand_for3.sav', 'trained_models/grad_bst3.sav']

logit = pickle.load(open(models[0], 'rb'))
dec_tree = pickle.load(open(models[1], 'rb'))
rand_for = pickle.load(open(models[2], 'rb'))
grad_bst = pickle.load(open(models[3], 'rb'))

precision = []
recall = []
balanced_accuracy_score = []
model_used = []


from sklearn import metrics

log_1 = logit.predict(X_train)
precision.append(metrics.precision_score(y_train, log_1))
recall.append(metrics.recall_score(y_train, log_1))
balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_train, log_1))
model_used.append("Logistic Regression")

dec_1 = dec_tree.predict(X_train)
precision.append(metrics.precision_score(y_train, dec_1))
recall.append(metrics.recall_score(y_train, dec_1))
balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_train, dec_1))
model_used.append("Decision Tree")

rf_1 = rand_for.predict(X_train)
precision.append(metrics.precision_score(y_train, rf_1))
recall.append(metrics.recall_score(y_train, rf_1))
balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_train, rf_1))
model_used.append("Random Forest")

grb_1 = grad_bst.predict(X_train)
precision.append(metrics.precision_score(y_train, grb_1))
recall.append(metrics.recall_score(y_train, grb_1))
balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_train, grb_1))
model_used.append("Gradient Boosting")

results_df = pd.DataFrame({"Model": model_used, "Recall":recall, "Precision" : precision,
"Balanced Accuracy" : balanced_accuracy_score})

results_df = results_df.sort_values(by = ['Recall', 'Precision',
'Balanced Accuracy'], ascending = False).reset_index(drop=True)

results_df.to_csv('trained_models/results_test3.csv', index = False)
results_df
cm_log1 = metrics.confusion_matrix(y_train, log_1)
cm_dec1 = metrics.confusion_matrix(y_train, dec_1)
cm_rf1 = metrics.confusion_matrix(y_train, rf_1)
cm_grb1 = metrics.confusion_matrix(y_train, grb_1)
cm_log1
# final_prediction = grad_bst.predict(X_test)
precision = []
recall = []
balanced_accuracy_score = []
model_used = []
precision.append(metrics.precision_score(y_train, final_prediction))
recall.append(metrics.recall_score(y_train, final_prediction))
balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_train, final_prediction))
model_used.append("Gradient Boosting")

final_results = pd.DataFrame({"Model": model_used, "Recall":recall, "Precision" : precision,
"Balanced Accuracy" : balanced_accuracy_score})
