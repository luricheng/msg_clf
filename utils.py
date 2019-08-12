def split_csv2train_test(csv, train_op):
    # 抽样比例100% 实现数据打乱
    shuffle_csv = csv.sample(frac=1.0, random_state=1)
    k = int(len(csv) * train_op)
    return shuffle_csv[: k], shuffle_csv[k:]


def __call_all(obj_list, func_name, x):
    for o in obj_list:
        if hasattr(o, func_name):
            x = getattr(o, func_name)(x)
        else:
            raise "obj: %s has not func: %s" % (o, func_name)
    return x


def fit_transform_all(transform_list, x):
    return __call_all(transform_list, "fit_transform", x)


def transform_all(transform_list, x):
    return __call_all(transform_list, "transform", x)


def get_class_name(model):
    return model.__class__.__name__
