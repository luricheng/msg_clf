from tqdm import tqdm
import time
import os
from sklearn import metrics
import numpy as np


def split_df2train_test(df, train_op):
    """
    split dataFrame to (train_dataFrame, test_dataFrame)
    :param df: dataFrame
    :param train_op: train set ratio
    :return: (dataFrame, dataFrame)
    """
    # 抽样比例100% 实现数据打乱
    shuffle_df = df.sample(frac=1.0, random_state=1)
    k = int(len(df) * train_op)
    return shuffle_df[: k], shuffle_df[k:]


def df2train_test_set(df, x_label, y_label, train_op=0.7, pre_processing=None):
    """
    parse dataFrame to (train_set, test_set)
    :param df: dataFrame
    :param x_label: x's column name in dataFrame
    :param y_label: y's column name in dataFrame
    :param train_op: train set ratio
    :param pre_processing: pipeline for data pre-processing
    :return:
    """
    train_df, test_df = split_df2train_test(df, train_op)
    print("train_set: %s, test_set: %s" % (len(train_df), len(test_df)))

    x_train, y_train = train_df[x_label].values, train_df[y_label].values.astype(int)
    x_test, y_test = test_df[x_label].values, test_df[y_label].values.astype(int)

    if pre_processing:
        x_train = pre_processing.fit_transform(x_train)
        x_test = pre_processing.transform(x_test)
    return (x_train, y_train), (x_test, y_test)


def get_class_name(model):
    return model.__class__.__name__


def train(train_set, test_set, models, output_dir):
    x_train, y_train = train_set
    x_test, y_test = test_set
    top_acc, top_model = -0.0, ""
    for model in tqdm(models):
        start_time = time.time()
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test).astype(int)
        acc = np.sum(y_pred == y_test) / len(y_test)
        model_name = get_class_name(model)
        if acc > top_acc:
            top_acc, top_model = acc, model_name
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)
        with open(os.path.join(output_dir, model_name + ".txt"), "w") as fp:
            output = "model:\n%s\n" \
                     "acc: %s\n" \
                     "confusion_matrix:\n%s\n" \
                     "classification_report:\n%s" \
                     "cost time: %s sec\n"
            output = output % (model, acc, metrics.confusion_matrix(y_true=y_test, y_pred=y_pred),
                               metrics.classification_report(y_pred=y_pred, y_true=y_test, target_names=['0', '1'],
                                                             digits=5),
                               time.time() - start_time)
            fp.write(output)
    print("top_acc: %s, model: %s" % (top_acc, top_model))
