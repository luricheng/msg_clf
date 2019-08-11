def split_csv2train_test(csv, train_op):
    # 抽样比例100% 实现数据打乱
    shuffle_csv = csv.sample(frac=1.0, random_state=1)
    k = int(len(csv) * train_op)
    return shuffle_csv[: k], shuffle_csv[k:]