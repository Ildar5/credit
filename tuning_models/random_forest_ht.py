from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import numpy as np
from tuning_models import gsearch as gs


def random_forest_tuning(x_train, y_train):

    max_features_range = np.arange(1, 6, 1)
    n_estimators_range = np.arange(10, 210, 10)
    param_grid = dict(max_features=max_features_range, n_estimators=n_estimators_range)

    rf = RandomForestClassifier()

    grid = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
    grid.fit(x_train, y_train)

    gs.best_params(grid)
    print('visualise:')
    gs.visualise(grid)
