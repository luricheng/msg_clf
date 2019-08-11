import pandas as pd
import utils
from model import tfidf, clfs
import numpy as np
from sklearn import metrics
import time

if __name__ == '__main__':
    csv_data = pd.read_csv('./data/train.csv', usecols=[0, 1])
    csv_data.loc[csv_data.Label == 'spam', 'Label'] = 1
    csv_data.loc[csv_data.Label == 'ham', 'Label'] = 0
    print("csv_data:\n", csv_data.head())

    train_csv, test_csv = utils.split_csv2train_test(csv_data, train_op=0.7)
    print("train_set: %s, test_set: %s" % (len(train_csv), len(test_csv)))

    for model_name, clf_model in clfs.models.items():
        start_time = time.time()
        x_train, y_train = train_csv.Text.values, train_csv.Label.values
        tfidf_model, tfidf_matrix = tfidf.train(x_train)
        clf_model.fit(tfidf_matrix, y_train)

        x_test, y_test = test_csv.Text.values, test_csv.Label.values
        y_pred = clf_model.predict(tfidf_model.transform(x_test))
        acc = np.sum(y_pred == y_test) / len(y_test)
        print("model: %s\nacc: %s" % (model_name, acc))
        print(metrics.classification_report(y_pred=y_pred.astype(int), y_true=y_test.astype(int), target_names=['0', '1']))
        print('cost time: %s' % (time.time() - start_time))

