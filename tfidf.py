from sklearn.feature_extraction.text import TfidfVectorizer


class Tfidf(TfidfVectorizer):
    def fit_transform(self, raw_documents, y=None):
        return super().fit_transform(raw_documents, y).todense()

    def transform(self, raw_documents, copy=True):
        return super().transform(raw_documents, copy=copy).todense()
