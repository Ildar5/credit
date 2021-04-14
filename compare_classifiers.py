from main import normalize_features
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import SGDClassifier
import seaborn as sns


if __name__ == '__main__':
    X, y = normalize_features()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=10)
    print(X_train.shape, y_train.shape)
    print(X_test.shape, y_test.shape)

    names = [
        "Nearest_Neighbors",
        "Linear_SVM",
        "Polynomial_SVM",
        "RBF_SVM",
        "Gaussian_Process",
        "Gradient_Boosting",
        "Decision_Tree",
        "Extra_Trees",
        "Random_Forest",
        "Neural_Net",
        "AdaBoost",
        "Naive_Bayes",
        "QDA",
        "SGD"
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025),
        SVC(kernel="poly", degree=3, C=0.025),
        SVC(kernel="rbf", C=1, gamma=2),
        GaussianProcessClassifier(1.0 * RBF(1.0)),
        GradientBoostingClassifier(n_estimators=100, learning_rate=1.0),
        DecisionTreeClassifier(max_depth=5),
        ExtraTreesClassifier(n_estimators=10, min_samples_split=2),
        RandomForestClassifier(max_depth=5, n_estimators=100),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(n_estimators=100),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
        SGDClassifier(loss="hinge", penalty="l2")
    ]

    scores = []
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        scores.append(score)

    print(scores)

    df = pd.DataFrame()
    df['name'] = names
    df['score'] = scores
    print(df)

    cm = sns.light_palette("green", as_cmap=True)
    s = df.style.background_gradient(cmap=cm)
    print(s)

    sns.set(style="whitegrid")
    ax = sns.barplot(y="name", x="score", data=df)

# "Nearest_Neighbors": 0.881176
# "Linear_SVM": 0.810162
# "Polynomial_SVM": 0.847770
# "RBF_SVM": 0.957191
# "Gaussian_Process",
# "Gradient_Boosting": 0.987998
# "Decision_Tree": 0.981396
# "Extra_Trees": 0.982597
# "Random_Forest": 0.961592
# "Neural_Net": 0.8407681536307261
# "AdaBoost": 0.984397
# "Naive_Bayes": 0.928386
# "QDA": 0.969194
# "SGD": 0.947189

# leaders
# "Gradient_Boosting": 0.987998 after tuning 0.98959
# "AdaBoost": 0.984397
# "Extra_Trees": 0.982597
# "Decision_Tree": 0.981396
