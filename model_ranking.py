# Function used to compare the performance of multiple models

def model_ranking(X, y):

    import pandas as pd
    import numpy as np
    from scipy import stats, sqrt
    from sklearn import metrics
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import cross_val_predict
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import RandomizedSearchCV
    import warnings
    import random
    random.seed(69)
    warnings.filterwarnings('ignore')

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 69)

    precision = []
    recall = []
    balanced_accuracy_score = []
    model_used = []

    from sklearn.linear_model import LogisticRegression
    logisticRegr = LogisticRegression()
    logisticRegr.fit(X_train, y_train)
    y_pred = cross_val_predict(logisticRegr, X_test, y_test, cv=6)
    precision.append(metrics.average_precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_test, y_pred))
    model_used.append("Logistic Regression")

    from sklearn.tree import DecisionTreeClassifier
    parameters_dt={ "criterion" : ['gini', 'entropy'], 'max_depth': range(5,20,5)}
    dtclass = GridSearchCV(DecisionTreeClassifier(), parameters_dt, cv=3, verbose=1, n_jobs=-1, scoring="f1")
    dtclass.fit(X_train, y_train)
    y_pred = cross_val_predict(dtclass, X_test, y_test, cv=6)
    precision.append(metrics.average_precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_test, y_pred))
    model_used.append("Decision Tree")

    from sklearn.ensemble import RandomForestClassifier
    parameters_rf = {  'n_estimators': [50, 100, 200, 500, 1000, 2000],
                       'max_features': ['auto', 'sqrt'],
                       'max_depth': [5, 10, 20, 50, 100],
                       'min_samples_split': [2, 5, 10, 40],
                       'min_samples_leaf': [1, 2, 5, 10, 20],
                       'bootstrap': [True, False]  }

    rfclass=RandomizedSearchCV(RandomForestClassifier(), parameters_rf, cv = 3, verbose=1, n_jobs = -1, scoring="f1")
    rfclass.fit(X_train, y_train)
    y_pred = cross_val_predict(rfclass, X_test, y_test, cv=6)
    precision.append(metrics.average_precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_test, y_pred))
    model_used.append("Random Forest")

    from sklearn.ensemble import GradientBoostingClassifier
    parameters_gb = {
            "loss":["deviance"],
            "min_samples_split": np.linspace(0.1, 0.5, 5),
            "min_samples_leaf": np.linspace(0.1, 0.5, 5),
            "max_depth":[3, 5, 8, 15],
            "max_features":["auto"],
            "criterion": ["friedman_mse"],
            "subsample":[1],
            "n_estimators":[50, 100, 200, 500]
        }
    gb_clf2 = GridSearchCV(GradientBoostingClassifier(), parameters_gb, cv=3, verbose=1, n_jobs=-1, scoring="f1")
    gb_clf2.fit(X_train, y_train)
    y_pred = cross_val_predict(gb_clf2, X_test, y_test, cv=6)
    precision.append(metrics.precision_score(y_test, y_pred))
    recall.append(metrics.recall_score(y_test, y_pred))
    balanced_accuracy_score.append(metrics.balanced_accuracy_score(y_test, y_pred))
    model_used.append("Gradient Boosting")

    results_df = pd.DataFrame({"Model": model_used, "Balanced Accuracy" : balanced_accuracy_score,
    "Precision" : precision, "Recall":recall})

    results_df = results_df.sort_values(by = ['Precision', 'Recall', 'Balanced Accuracy'], ascending = False).reset_index(drop=True)

    return results_df, logisticRegr, dtclass, rfclass, gb_clf2
