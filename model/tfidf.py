from sklearn.feature_extraction.text import TfidfVectorizer


def train(x):
    model = TfidfVectorizer().fit(x)
    sparse_result = model.transform(x)
    return model, sparse_result.todense()


def predict(model, x):
    return model.transform(x).todense()