from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer_names
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def cv_plot(cv_range, mean_error, std_error, hyper_param):
    plt.errorbar(cv_range, mean_error, yerr=std_error, linewidth=3)
    plt.xlabel(hyper_param)
    plt.ylabel('F1 weighted')
    plt.show()


def min_df_cv(min_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=min_df_val, max_df=100)


def max_df_cv(max_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.01, max_df=max_df_val)


def ngram_cv(ngram_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.01, max_df=100,
                           ngram_range=(1, ngram_val))


def vectoriser_cv(hyper_param, text, cv_range):
    if hyper_param == "min df":
        get_vectoriser = min_df_cv
    elif hyper_param == "max df":
        get_vectoriser = max_df_cv
    else:
        get_vectoriser = ngram_cv
    mean_error = []
    std_error = []
    for i in cv_range:
        vectoriser = get_vectoriser(i)
        X = vectoriser.fit_transform(text)
        model = KNeighborsClassifier(n_neighbors=25)
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    cv_plot(cv_range, mean_error, std_error, hyper_param)


def k_cv(text):
    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.01, max_df=100,
                                 ngram_range=(1, 4))
    X = vectorizer.fit_transform(text)

    n_neighbours = []
    mean_error = []
    std_error = []
    # loop through k range
    for k in range(1, 100, 5):
        # intialise model
        model = KNeighborsClassifier(n_neighbors=k)
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
        n_neighbours.append(k)
    cv_plot(n_neighbours, mean_error, std_error, 'k')


df = pd.read_csv("dataset.tsv", sep="\t")
text = df.iloc[:, -1]
others = df.iloc[:, 2:7]
y = df.iloc[:, 1]

vectoriser_cv('min df', text, [0.0001, 0.001, 0.01, 0.1, 1])
vectoriser_cv('max df', text, [10, 50, 100, 200, 400])
vectoriser_cv('ngram', text, [1, 2, 3, 4, 5, 6, 7, 8, 9])
k_cv(text)

vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.01, max_df=100,
                             ngram_range=(1, 4))
X = vectoriser.fit_transform(text)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

kNN_model = KNeighborsClassifier(n_neighbors=25)
kNN_model.fit(xtrain, ytrain)
predictions = kNN_model.predict(xtest)
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(xtrain, ytrain)
predictions = dummy.predict(xtest)
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))
