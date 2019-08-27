from transform_base import TransformBase
import jieba


class CnCut(TransformBase):
    def fit_transform(self, raw_documents, y=None):
        return self.transform(raw_documents)

    def transform(self, raw_documents, copy=True):
        return [" ".join(jieba.lcut(x)) for x in raw_documents]


if __name__ == '__main__':
    a = CnCut()
    print(a.transform(["香港记者跑得真快"]))
