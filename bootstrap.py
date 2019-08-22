import pandas as pd
import utils
import numpy as np
from sklearn import metrics
import time
from sklearn.decomposition import PCA
from tfidf import Tfidf
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import os
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier

# todo Label==1的样本 数量过少
# todo 后续: 过采样???

OUTPUT_DIR = "./output"

transform_list = [
    Tfidf(),
    StandardScaler(),
    PCA(n_components=148, random_state=1),
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

    top_acc, top_model = -0.0, ""
    for model in tqdm(models):
        start_time = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).astype(int)
        acc = np.sum(y_pred == y_test) / len(y_test)
        model_name = utils.get_class_name(model)
        if acc > top_acc:
            top_acc, top_model = acc, model_name
        if not os.path.exists(OUTPUT_DIR):
            os.mkdir(OUTPUT_DIR)
        with open(os.path.join(OUTPUT_DIR, model_name + ".txt"), "w") as fp:
            output = "model:\n%s\n" \
                     "acc: %s\n" \
                     "confusion_matrix:\n%s" \
                     "classification_report:\n%s" \
                     "cost time: %s sec\n"
            output = output % (model, acc, metrics.confusion_matrix(y_true=y_test, y_pred=y_pred),
                               metrics.classification_report(y_pred=y_pred, y_true=y_test, target_names=['0', '1'],
                                                             digits=5),
                               time.time() - start_time)
            fp.write(output)
    print("top_acc: %s, model: %s" % (top_acc, top_model))
