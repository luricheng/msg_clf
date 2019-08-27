import pandas as pd
import utils
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from cn_cut import CnCut
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sparse2dense import Sparse2Dense

"""
中文文言文/白话文分类
"""

OUTPUT_DIR = "./output"

pre_processing = Pipeline([
    ("cut", CnCut()),                                           # 中文分词
    ("tf-idf", TfidfVectorizer(token_pattern=r"(?u)\b\w+\b")),  # 默认的token_pattern只保留长度>=2的词 但是针对中文不合适 此处保留>=1
    ('dense', Sparse2Dense()),
    ("stand", StandardScaler()),
    ("pca", PCA(n_components=148, random_state=1)),
])

models = [
    XGBClassifier(random_state=1, n_estimators=100, n_jobs=4),
    RandomForestClassifier(random_state=1, n_estimators=100, n_jobs=4),
    SVC(random_state=1, kernel='rbf', class_weight='balanced'),
    SGDClassifier(random_state=1, n_jobs=4)
]

if __name__ == '__main__':
    csv_data = pd.read_csv("./data/train.txt", usecols=[1, 2])
    print(csv_data.head())

    train, test = utils.df2train_test_set(csv_data, "text", "y", 0.7, pre_processing)

    utils.train(train, test, models, OUTPUT_DIR)
