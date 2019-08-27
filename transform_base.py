
class TransformBase(object):
    def fit(self, x, y=None):
        raise NotImplementedError

    def fit_transform(self, x, y=None):
        raise NotImplementedError

    def transform(self, x, copy=True):
        raise NotImplementedError
