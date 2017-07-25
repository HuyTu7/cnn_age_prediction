# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import re
import pickle

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from pyvi.pyvi import ViTokenizer, ViPosTagger
from sklearn.linear_model import SGDClassifier

from younet_rnd_infrastructure.tri.common import utils as yn_utils
from younet_rnd_infrastructure.tri.machine_learning.preprocessing import preprocessor
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
np.random.seed(seed)


from experiment.utils import utils


class MyCountVectorizer(CountVectorizer):
    def __init__(self):
        super(self)

    def transform(self, raw_documents):
        return self.transform(raw_documents).to_array()

SOURCE_URL = './../../crawl_info/user_post/output/posts_user.csv'
SOURCE_SAMPLE_URL = './input/sample_5000.csv'
SOURCE_DIR = './../../crawl_info/user_post/'


def my_tokenize(posts_str):
    posts = posts_str.split('(^-^)')

    tokens = []
    for post in posts:
        post = utils.icons(post)
        tokens.extend(ViTokenizer.tokenize(post).split(' '))
    return tokens


def my_preprocessor(posts_str):
    result = re.sub('[?|$|.|!|&|%|:|;|0-9]', '', posts_str) #re.sub(re.compile(reg), '', posts_str)
    result = result.lower()
    # p = preprocessor.Preprocessor([posts_str])
    # result = p.preprocess()[0]
    return result

def data_load():
    train_df = pd.read_csv(SOURCE_DIR + 'post_user_from_04_2017.csv').iloc[:1000, :]
    train_df = train_df.dropna()
    test_df = pd.read_csv(SOURCE_DIR + 'post_user_from_04_2017.csv').iloc[1000:1200, :]
    test_df = test_df.dropna()

    train_df = utils.add_age_category_to_df(train_df)
    test_df = utils.add_age_category_to_df(test_df)

    train_df['post'] = map(lambda x: x.decode('utf-8'), train_df['post'])
    test_df['post'] = map(lambda x: x.decode('utf-8'), test_df['post'])

    train_df['post'] = map(lambda x: re.sub('[?|$|.|!|&|%|:|;|0-9]', '', x), train_df['post'])
    test_df['post'] = map(lambda x: re.sub('[?|$|.|!|&|%|:|;|0-9]', '', x), test_df['post'])

    train_df.to_csv('./train1.csv', index=None)
    test_df.to_csv('./test1.csv', index=None)

def data_preprocess(df, filename, o):
    vectorize = CountVectorizer(tokenizer=my_tokenize, lowercase=True, min_df=0.007, ngram_range=(1, 1))
    vectorized_df = vectorize.fit_transform(df)
    scaled_df = preprocessing.MaxAbsScaler().fit_transform(vectorized_df)
    #np.savetxt(filename, scaled_df)

    ''''''
    PIK = filename
    with open(PIK, "wb") as f:
        pickle.dump(len(data), f)
        for value in data:
            pickle.dump(value, f)

    with open(PIK, "rb") as f:
        for _ in range(pickle.load(f)):
            data2.append(pickle.load(f))
    print data2


    f = open(filename, 'w')
    pickle.dump(scaled_df, f)
    f.close()

    f2 = open(filename, 'r')
    s = pickle.load(filename)
    f2.close()
    df = pd.DataFrame(
        {'v_text': s,
         })
    df.to_csv(o, index=None)
    '''


def train_on_server():
    # posts = train_df.loc[:, 'post']
    train_df = pd.read_csv('./train1.csv')
    test_df = pd.read_csv('./test1.csv')

    '''
    estimators = [('vectorizer', vectorize),
                  ('scaler', preprocessing.MaxAbsScaler()),
                  # ('clf', SGDClassifier(loss="hinge", penalty="l2"))
                  # ('clf', SVC(C=1, kernel='linear'))
                  ('clf', LinearSVC(C=50, random_state=7))]

    clf = Pipeline(estimators)

    clf = GridSearchCV(estimator=clf,
                       param_grid=dict(
                           clf__C=[10],
                           vectorizer__min_df=[0.007],
                           vectorizer__ngram_range=[(1, 2), (1, 3)]),
                       n_jobs=-1, cv=4, verbose=10, scoring='f1_macro')
    '''
    X = train_df.loc[:, 'post']
    y = train_df.loc[:, 'age_category']
    X_test = test_df.loc[:, 'post']
    y_test = test_df.loc[:, 'age_category']

    data_preprocess(X, './train_df.pickle', './train10.csv')
    data_preprocess(X_test, './test_df.pickle', './test10.csv')
    '''
    print 'Size train: %s' % X.shape
    print 'Size test: %s' % X_test.shape
    clf.fit(X, y)
    print 'Done'

    print 'Dummy classification model'
    pickle.dump(clf, open('./output/clf_model.pkl', 'wb'))
    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)

    print("Best score found on development set:")
    print()
    print(clf.best_score_)

    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()
    print()
    print("Scores on test set: %s" % classification_report(y_test, clf.predict(X_test)))
    print()
    '''

if __name__ == '__main__':
    #data_load()
    train_on_server()
