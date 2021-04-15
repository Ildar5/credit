from tuning_models import gradient_boosting_ht as gb, ada_boost_ht as ab, design_tree_ht as dt
from main import get_train_test_split


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = get_train_test_split()

    # print('design_tree_tuning:')
    dt.design_tree_tuning(X_train, y_train)
    # The best parameters are {'max_depth': 12, 'max_features': 5} with a score of 0.983
    dt.check_overfitting()
    # Identify overfitting shows that it tends to be overfitted after max_depth = 6,
    # best params is max_depth = 9 and max_features = 4

    print('gradient_boosting_tuning:')
    gradient_boosting_best_params = gb.gradient_boosting_tuning(X_train, y_train)
    print('gradient_boosting_best_params:')
    print(gradient_boosting_best_params)
    gb.check_overfitting()
    # Identify overfitting shows that it tends to be overfitted after max_depth = 3,
    # best params is max_depth = 3 and max_features = 4
    #
    print('ada_boost_tuning:')
    ada_boost_best_params = ab.ada_boost_tuning(X_train, y_train)
    print('>%s %.3f (%.3f)' % ada_boost_best_params)
    # # best params are max_depth=15 and learning_rate=1

