from sklearn.model_selection import GridSearchCV
import numpy as np
from tuning_models import gsearch as gs
from main import get_train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot as plt


def design_tree_tuning(x_train, y_train):

    max_features_range = np.arange(1, 6, 1)
    max_depth_range = np.arange(1, 21, 1)
    param_grid = dict(max_features=max_features_range, max_depth=max_depth_range)

    rf = DecisionTreeClassifier()

    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    gs.dt_best_params(grid)
    print('visualise:')
    gs.dt_visualise(grid)


def check_overfitting():
    x_train, x_test, y_train, y_test = get_train_test_split()
    # define the tree depths to evaluate
    depths = [i for i in range(1, 21)]
    max_features = [i for i in range(1, 7)]
    for max_feature in max_features:
        print('max_feature:{}' . format(max_feature))
        # define lists to collect scores
        train_scores, test_scores = list(), list()
        # evaluate a decision tree for each depth
        for depth in depths:
            # configure the model
            model = DecisionTreeClassifier(max_depth=depth, max_features=max_feature)
            # fit model on the training dataset
            model.fit(x_train, y_train)
            # evaluate on the train dataset
            train_yhat = model.predict(x_train)
            train_acc = accuracy_score(y_train, train_yhat)
            train_scores.append(train_acc)
            # evaluate on the test dataset
            test_yhat = model.predict(x_test)
            test_acc = accuracy_score(y_test, test_yhat)
            test_scores.append(test_acc)
            # summarize progress
            print('depth => %d, train: %.3f, test: %.3f' % (depth, train_acc, test_acc))
        # plot of train and test scores vs tree depth
        plt.plot(depths, train_scores, '-o', label='Train')
        plt.plot(depths, test_scores, '-o', label='Test')
        plt.legend()
        plt.title('max_features:{}'.format(max_feature), loc='right')
        plt.savefig('scoring_models/plot_images/overfitting_check/dt/decision_tree_max_features_{}.png'.format(max_feature))

