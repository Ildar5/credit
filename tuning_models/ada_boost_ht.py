from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np


# get a list of models to evaluate
def get_ada_boost_depths_models():
    models = dict()
    # explore depths from 1 to 20
    for i in range(1, 21):
        # define base model
        base = DecisionTreeClassifier(max_depth=i)
        # define ensemble model
        models[str(i)] = AdaBoostClassifier(base_estimator=base)
    return models


# get a list of models to evaluate
def get_ada_boost_lr_models():
    models = dict()
    # explore learning rates from 0.1 to 2 in 0.1 increments
    for i in np.arange(0.1, 2.1, 0.1):
        key = '%.3f' % i
        base = DecisionTreeClassifier(max_depth=15)
        models[key] = AdaBoostClassifier(base_estimator=base, learning_rate=i)
    return models


# evaluate a given model using cross-validation
def evaluate_ada_boost_model(model, X, y):
    # define the evaluation procedure
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=10)
    # evaluate the model and collect the results
    scores = cross_val_score(model, X, y, scoring='accuracy', cv=cv, n_jobs=-1)
    return scores


def ada_boost_tuning(x_train, y_train):
    # best ones starts from 15
    models = get_ada_boost_depths_models()
    # 1 is fine
    # models = get_ada_boost_lr_models()
    # evaluate the models and store results
    results, names, all_scores = list(), list(), list()
    for name, model in models.items():
        # evaluate the model
        scores = evaluate_ada_boost_model(model, x_train, y_train)
        # store the results
        results.append(scores)
        names.append(name)
        # summarize the performance along the way
        print('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
        all_scores.append('>%s %.3f (%.3f)' % (name, np.mean(scores), np.std(scores)))
    return all_scores

