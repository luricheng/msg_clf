from transform_base import TransformBase


class Sparse2Dense(TransformBase):
    def fit(self, x, y=None):
        pass

    def transform(self, x, y=None):
        return x.todense()

    def fit_transform(self, x, y=None):
        return x.todense()
