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
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def cv_plot(cv_range, mean_error, std_error, hyper_param, log):
    plt.errorbar(cv_range, mean_error, yerr=std_error, linewidth=3)
    if log:
        plt.xscale("log")
    plt.xlabel(hyper_param)
    plt.ylabel('F1 weighted')
    plt.show()


def min_df_cv(min_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=min_df_val, max_df=100)


def max_df_cv(max_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.1, max_df=max_df_val)


def ngram_cv(ngram_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.1, max_df=100,
                           ngram_range=(1, ngram_val))


def vectoriser_cv(hyper_param, text, cv_range, votes):
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
        X = pd.concat([pd.DataFrame(X.toarray()), votes], axis=1)
        scale = StandardScaler(with_mean=False)
        scale.fit_transform(X)
        model = LinearSVC()
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    cv_plot(cv_range, mean_error, std_error, hyper_param, False)

    print(cv_range)
    print(mean_error)
    print(std_error)



def penalty_cv(text):
    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.01, max_df=100,
                                 ngram_range=(1, 2))
    X = vectorizer.fit_transform(text)
    c_range = [0.00000000001, 0.0000000001, 0.000000001, 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000, 100000000000, 1000000000000, 10000000000000, 100000000000000]
    mean_error = []
    std_error = []

    for c in c_range:
        model = LinearSVC(C=c)
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    cv_plot(c_range, mean_error, std_error, 'k', True)

    print(c_range)
    print(mean_error)
    print(std_error)


df = pd.read_csv("dataset.tsv", sep="\t")
text = df.content.astype("U")
others = df.iloc[:, 2:22]
y = df.rating

vectoriser_cv('min df', text, [0.0001, 0.001, 0.01, 0.1, 1], others)
vectoriser_cv('max df', text, [50, 100, 500, 1000], others)
vectoriser_cv('ngram', text, [1, 2, 3, 4, 5, 6, 7, 8, 9], others)
penalty_cv(text)


vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.1, max_df=100,
                             ngram_range=(1, 2))


X = vectoriser.fit_transform(text)
X = pd.concat([pd.DataFrame(X.toarray()), others], axis=1)
scale = StandardScaler(with_mean=False)
scale.fit_transform(X)
xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)


svm_model = LinearSVC(C=1)
svm_model.fit(xtrain, ytrain)
predictions = svm_model.predict(xtest)
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))

dummy = DummyClassifier(strategy='uniform')
dummy.fit(xtrain, ytrain)
predictions = dummy.predict(xtest)
print(classification_report(ytest, predictions))
print(confusion_matrix(ytest, predictions))
