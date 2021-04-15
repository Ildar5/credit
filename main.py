from sklearn import preprocessing
from sklearn.metrics import plot_confusion_matrix
from plot_metric.functions import BinaryClassification
from filter.feature_selector import FeatureSelector
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
import time


def get_dataset():
    return pd.read_csv('dataset/data.csv', sep=';')


def features_and_response(dataset):
    # ds = get_dataset()
    response = dataset['expert']
    features = dataset.drop('expert', axis=1)
    return features, response


def normalize_features():
    features = get_dataset()
    features['order_date'] = pd.to_datetime(features['order_date']).astype(np.int64) / 10 ** 9
    # we can remove missing column values, there are 19 of then in dataset
    # features.dropna(inplace=True)
    # another way is filling missing values
    features.fillna(features.mean(), inplace=True)
    print('dataset count')
    print(len(features))

    x, y = features_and_response(features)

    # filter less important features which was figured out from feature select functions
    x.drop([
        'closed_creds',
        'active_cred_max_overdue',
        'region',
        'month_income',
        'active_cred_day_overdue',
        'active_cred_sum_overdue',
        'gender'
    ], axis=1, inplace=True)

    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)
    x = pd.DataFrame(x_scaled)

    return x, y


def get_train_test_split():
    x, y = normalize_features()
    return train_test_split(x, y, test_size=0.1, random_state=10)


def confusion_matrix(name, model, x_test, y_test):
    class_names = ['approved', 'rejected']
    # Plot non-normalized confusion matrix
    titles_options = [("{} Confusion matrix before normalisation".format(name), None),
                      ("{} Normalised confusion matrix".format(name), 'true')]
    for title, normalize in titles_options:
        disp = plot_confusion_matrix(
            model, x_test, y_test,
            display_labels=class_names,
            cmap=plt.cm.Blues,
            normalize=normalize
        )
        disp.ax_.set_title(title)

        write_scoring_results(name, title)
        write_scoring_results(name, disp.confusion_matrix)
        print(title)
        print(disp.confusion_matrix)

    plt.savefig('scoring_models/plot_images/confusion_matrix/confusion_matrix_{}.png'.format(name))


def roc_curve_plot(name, y_pred, y_test):
    # Visualisation with plot_metric
    bc = BinaryClassification(y_test, y_pred, labels=["ROC кривая", "Случайный выбор"])
    # Figures
    plt.figure(figsize=(10, 10))
    bc.plot_roc_curve()
    plt.savefig('scoring_models/plot_images/roc_curve/roc_curve_{}.png'.format(name))
    # plt.show()


def write_scoring_results(model_name, text):
    file = open("scoring_models/{}.txt".format(model_name), "a")
    file.write(str(text))
    file.write('\n\n')
    file.close()


def write_params_to_file(name, time, score, f1_s, recall, precision):
    remove_files(name)
    write_scoring_results(name, 'train time: {}'.format(time))
    write_scoring_results(name, 'score: {}'.format(score))
    write_scoring_results(name, 'f1 score: {}'.format(f1_s))
    write_scoring_results(name, 'recall: {}'.format(recall))
    write_scoring_results(name, 'precision: {}'.format(precision))


def remove_files(model_name):
    path = 'scoring_models/{}.txt'.format(model_name)
    if os.path.isfile(path):
        os.remove(path)


def get_feature_selector():
    features = get_dataset()
    features['order_date'] = pd.to_datetime(features['order_date']).astype(np.int64) / 10 ** 9
    train_labels = features['expert']
    train = features.drop(columns=['expert'])

    return FeatureSelector(data=train, labels=train_labels)


def missing_features(fs):
    fs.identify_missing(missing_threshold=0.6)
    missing_features = fs.ops['missing']
    return missing_features[:10]


def get_scores(y_test, y_pred):
    f1_s = f1_score(y_test, y_pred, average='macro')
    recall = recall_score(y_test, y_pred, average='binary')
    precision = precision_score(y_test, y_pred, average='binary')

    return f1_s, recall, precision


def best_model():
    dt = DecisionTreeClassifier(max_depth=9, max_features=4)
    dt.fit(X_train, y_train)
    return dt


if __name__ == '__main__':

    X_train, X_test, y_train, y_test = get_train_test_split()
    # Decision Tree after normalisation and tuning
    start_train = time.time()
    dt = best_model()
    end_train = time.time()

    score = dt.score(X_test, y_test)
    y_pred = dt.predict(X_test)

    f1_s, recall, precision = get_scores(y_test, y_pred)
    print('train time: {}'.format(end_train - start_train))
    print('f1_s: {}'.format(f1_s))
    print('recall: {}'.format(recall))
    print('precision: {}'.format(precision))

    write_params_to_file('Final Decision Tree', end_train - start_train, score, f1_s, recall, precision)
    confusion_matrix('Final Decision Tree', dt, X_test, y_test)
    roc_curve_plot('Final Decision Tree', y_pred, y_test)
