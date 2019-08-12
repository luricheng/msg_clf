import pandas as pd
import utils
import numpy as np
from sklearn import metrics
import time
from sklearn.decomposition import PCA
from Tfidf import Tfidf
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# todo Label==1的样本 数量过少
# todo 后续: 过采样???

transform_list = [
    Tfidf(),
    PCA(n_components=100),
]

models = [
    XGBClassifier(random_state=1, n_estimators=100, n_jobs=4),
    RandomForestClassifier(random_state=1, n_estimators=100, n_jobs=4),
    SVC(random_state=1, kernel='rbf', class_weight='balanced'),
    SGDClassifier(random_state=1, n_jobs=4)
]

if __name__ == '__main__':
    csv_data = pd.read_csv('./data/train.csv', usecols=[0, 1])
    csv_data.loc[csv_data.Label == 'spam', 'Label'] = 1
    csv_data.loc[csv_data.Label == 'ham', 'Label'] = 0
    print("csv_data:\n", csv_data.head())

    train_csv, test_csv = utils.split_csv2train_test(csv_data, train_op=0.7)
    print("train_set: %s, test_set: %s" % (len(train_csv), len(test_csv)))

    x_train, y_train = train_csv.Text.values, train_csv.Label.values.astype(int)
    x_test, y_test = test_csv.Text.values, test_csv.Label.values.astype(int)

    x_train = utils.fit_transform_all(transform_list, x_train)
    x_test = utils.transform_all(transform_list, x_test)

    for model in models:
        start_time = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = np.sum(y_pred == y_test) / len(y_test)
        print("====> model: %s" % utils.get_class_name(model))
        print("====> acc: %s" % acc)
        print('====> cost time: %s' % (time.time() - start_time))
        print(metrics.classification_report(y_pred=y_pred.astype(int),
                                            y_true=y_test.astype(int),
                                            target_names=['0', '1']))

