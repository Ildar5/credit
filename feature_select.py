import main as m
import pandas as pd

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

fs = m.get_feature_selector()


def missing_features_detect():
    missing_features = m.missing_features(fs)
    if missing_features:
        # fs.plot_missing()
        print('missing features')
        print(missing_features)
        print(fs.missing_stats.head(10))
    else:
        print('no missing features')


def find_single_unique_features():
    fs.identify_single_unique()
    single_unique = fs.ops['single_unique']
    if single_unique:
        print('single unique')
        print(single_unique)
        # fs.plot_unique()
        fs.unique_stats.sample(5)
    else:
        print('no single unique')


def find_correlated_features():
    fs.identify_collinear(correlation_threshold=0.85)
    correlated_features = fs.ops['collinear']
    if correlated_features:
        print('correlated features')
        print(correlated_features[:5])
        # fs.plot_collinear()
    else:
        print('no correlated features')


def identify_zero_importance():
    fs.identify_zero_importance(
        task='classification',
        eval_metric='auc',
        n_iterations=10,
        early_stopping=True)

    one_hot_features = fs.one_hot_features
    base_features = fs.base_features
    print('There are %d original features' % len(base_features))
    print('There are %d one-hot features' % len(one_hot_features))


def zero_importance_features():
    fs.data_all.head(10)
    zero_importance_features = fs.ops['zero_importance']
    print('zero_importance_features')
    print(zero_importance_features)

    fs.plot_feature_importances(threshold=0.99, plot_n=16)


def feature_importances():
    print('fs.feature_importances')
    print(fs.feature_importances)


def identify_low_importance():
    fs.identify_low_importance(cumulative_importance=0.75)
    low_importance_features = fs.ops['low_importance']
    print('low_importance_features')
    print(low_importance_features[:5])


if __name__ == '__main__':

    missing_features_detect()
    find_single_unique_features()
    find_correlated_features()
    identify_zero_importance()
    zero_importance_features()
    feature_importances()
    identify_low_importance()

    # we will remove less important features one's that is not required by task,
    # sush as:
    # closed_creds, active_cred_max_overdue,
    # region, month_income, active_cred_day_overdue,
    # active_cred_sum_overdue, gender
