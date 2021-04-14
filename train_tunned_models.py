from sklearn.model_selection import train_test_split
from main import normalize_features, confusion_matrix, roc_curve_plot, get_scores, write_params_to_file
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import time


if __name__ == '__main__':

    X, y = normalize_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)

    # RandomForest
    start_train = time.time()
    rfc = RandomForestClassifier(max_features=5, n_estimators=110)
    rfc.fit(X_train, y_train)
    end_train = time.time()
    score = rfc.score(X_test, y_test)
    y_pred = rfc.predict(X_test)

    f1_s, recall, precision = get_scores(y_test, y_pred)

    write_params_to_file('Random Forest', end_train - start_train, score, f1_s, recall, precision)

    confusion_matrix('Random Forest', rfc, X_test, y_test)
    roc_curve_plot('Random Forest', y_pred, y_test)

    # GradientBoosting
    start_train = time.time()
    gbc = GradientBoostingClassifier(n_estimators=110, learning_rate=1)
    gbc.fit(X_train, y_train)
    end_train = time.time()
    score = gbc.score(X_test, y_test)
    y_pred = gbc.predict(X_test)

    f1_s, recall, precision = get_scores(y_test, y_pred)

    write_params_to_file('Gradient Boosting', end_train - start_train, score, f1_s, recall, precision)

    confusion_matrix('Gradient Boosting', gbc, X_test, y_test)
    roc_curve_plot('Gradient Boosting', y_pred, y_test)

    # AdaBoost
    start_train = time.time()
    base = DecisionTreeClassifier(max_depth=15)
    abc = AdaBoostClassifier(base_estimator=base, learning_rate=1)
    abc.fit(X_train, y_train)
    end_train = time.time()
    scores = abc.score(X_test, y_test)
    y_pred = abc.predict(X_test)

    f1_s, recall, precision = get_scores(y_test, y_pred)

    write_params_to_file('Ada Boost', end_train - start_train, score, f1_s, recall, precision)

    confusion_matrix('Ada Boost', abc, X_test, y_test)
    roc_curve_plot('Ada Boost', y_pred, y_test)
