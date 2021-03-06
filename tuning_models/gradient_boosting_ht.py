from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier
from main import get_train_test_split
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt


def gradient_boosting_tuning(x_train, y_train):
    param_test = {'n_estimators': range(10, 210, 10)}
    grid_search = GridSearchCV(
        estimator=GradientBoostingClassifier(
            learning_rate=0.1,
            min_samples_split=round(0.01 * len(x_train)),
            min_samples_leaf=50,
            max_depth=20,
            max_features=5,
            subsample=0.8,
            random_state=10
        ),
        param_grid=param_test,
        scoring='roc_auc',
        n_jobs=4,
        cv=5
    )

    grid_search.fit(x_train, y_train)

    return grid_search.best_params_, grid_search.best_score_


def check_overfitting():
    x_train, x_test, y_train, y_test = get_train_test_split()
    # define the tree depths to evaluate
    depths = [i for i in range(2, 21)]
    max_features = [i for i in range(2, 7)]
    for max_feature in max_features:
        print('max_feature:{}' . format(max_feature))
        # define lists to collect scores
        train_scores, test_scores = list(), list()
        # evaluate a decision tree for each depth
        for depth in depths:
            # configure the model
            model = GradientBoostingClassifier(max_depth=depth, max_features=max_feature)
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
        plt.show()
        plt.savefig('scoring_models/plot_images/overfitting_check/gb/gradient_boosting_max_features_{}.png'.format(max_feature))