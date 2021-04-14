from tuning_models import random_forest_ht as rf, ada_boost_ht as ab, gradient_boosting_ht as gb
from main import get_train_test_split


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = get_train_test_split()

    print('random_forest_tuning:')
    rf.random_forest_tuning(X_train, y_train)
    # The best parameters are {'max_features': 5, 'n_estimators': 180} with a score of 0.989641
    # but 'max_features': 5, 'n_estimators': 30 also good 0.989174

    print('gradient_boosting_tuning:')
    gradient_boosting_best_params = gb.gradient_boosting_tuning(X_train, y_train)
    print('gradient_boosting_best_params:')
    print(gradient_boosting_best_params)
    # # best one is ({'n_estimators': 110}, 0.997773772727393)
    #
    print('ada_boost_tuning:')
    ada_boost_best_params = ab.ada_boost_tuning(X_train, y_train)
    print('>%s %.3f (%.3f)' % ada_boost_best_params)
    # # best params are max_depth=15 and learning_rate=1
