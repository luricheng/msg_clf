from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
import numpy as np

models = {
    # 'rf': RandomForestClassifier(),
    'xgb': XGBClassifier(),
    # 'svm': SVC()
}


def fit_eva_all(x_train, x_test, y_train, y_test):
    pass


if __name__ == '__main__':
    m = XGBClassifier()
    x, y = np.array([1, 2, 3]).reshape(3, 1), np.array([3, 2, 1])
    m.fit(x, y)
    print(m.predict(x))
