from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier


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

