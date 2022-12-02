from itertools import cycle
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn import svm, datasets
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
#from sklearn.metrics import get_scorer_names, auc, roc_curve, roc_auc_score
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from sklearn.metrics import auc, roc_curve, roc_auc_score

nltk.download('stopwords')


def cv_plot(cv_range, mean_error, std_error, hyper_param, title, log):
    plt.title(title)
    if log:
        plt.xscale("log")
    plt.errorbar(cv_range, mean_error, yerr=std_error, linewidth=2)
    plt.xlabel(hyper_param)
    plt.ylabel('F1 macro')
    plt.show()


def min_df_cv(min_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=min_df_val)


def max_df_cv(max_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=10, max_df=max_df_val)


def ngram_cv(ngram_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=10, max_df=200,
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
        print(i)
        vectoriser = get_vectoriser(i)
        X = vectoriser.fit_transform(text)
        #X = pd.concat([pd.DataFrame(X.toarray()), votes], axis=1)
        model = MLPClassifier(hidden_layer_sizes=(50), max_iter=10000, verbose=True)
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    cv_plot(cv_range, mean_error, std_error, hyper_param, hyper_param + " cross validation", False)


def c_cv(text, votes):
    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=40, max_df=400,
                             ngram_range=(1, 3))
    X = vectorizer.fit_transform(text)
    #X = pd.concat([pd.DataFrame(X.toarray()), votes], axis=1)
    mean_error = []
    std_error = []
    c_range = [0.01, 0.05, 0.1, 0.5, 1, 5, 10, 100, 1000]
    # loop through k range
    for c in c_range:
        # intialise model
        model = MLPClassifier(hidden_layer_sizes=5, alpha=1.0/c, max_iter=10000, verbose=True)
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    cv_plot(c_range, mean_error, std_error, 'C', 'C values cross validation', True)


def nodes_cv(text, votes):
    vectorizer = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=40, max_df=400,
                             ngram_range=(1, 3))
    X = vectorizer.fit_transform(text)
    #X = pd.concat([pd.DataFrame(X.toarray()), votes], axis=1)
    hidden_layer_range = [1, 5, 10, 25, 50, 75, 100]
    mean_error = []
    std_error = []
    # loop through k range
    for i in hidden_layer_range:
        # intialise model
        model = MLPClassifier(hidden_layer_sizes=i, max_iter=10000, verbose=True)
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    cv_plot(hidden_layer_range, mean_error, std_error, 'Number of nodes in hidden layer', 'Hidden layer nodes cross validation', False)


def multiclass_roc_curves(X, Y, clf):
    classes = ['G', 'PG', 'PG-13', 'R', 'NC-17']
    Y = label_binarize(Y, classes=classes)
    N = len(classes)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3)
    try:
        # Dummy
        Ysc = np.array(clf.fit(Xtr, Ytr).predict_proba(Xte))[:, :, 1].transpose()
    except:
        # MLP
        Ysc = np.array(clf.fit(Xtr, Ytr).predict_proba(Xte))
    fpr, tpr = {}, {}
    for i in range(N):
        print(Yte.shape, Ysc.shape)
        fpr[i], tpr[i], _ = roc_curve(Yte[:, i], Ysc[:, i])
    fpr['micro'], tpr['micro'], _ = roc_curve(Yte.ravel(), Ysc.ravel())
    # First aggregate all false positive rates
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(N)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(N):
        mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= N

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    return fpr['macro'], tpr['macro'], roc_auc_score(Yte, Ysc, average='macro', multi_class='ovr')


df = pd.read_csv("dataset.tsv", sep="\t")
text = df.content.astype("U")
others = df.iloc[:, 2:22]
y = df.rating

vectoriser_cv('min df', text, [1, 5, 10, 20, 30, 40, 50], others)
vectoriser_cv('max df', text, [50, 100, 200, 300, 400, 500, 1000, 1500, 2000], others)
vectoriser_cv('ngram', text, [1, 2, 3, 4, 5], others)
nodes_cv(text, others)
c_cv(text, others)

combined_vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=10, max_df=200,
                                 ngram_range=(1, 4))

combined_X = combined_vectoriser.fit_transform(text)
combined_X = pd.concat([pd.DataFrame(combined_X.toarray()), others], axis=1)
text_vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=40, max_df=400,
                             ngram_range=(1, 3))
text_X = text_vectoriser.fit_transform(text)

combined_mlp_model = MLPClassifier(hidden_layer_sizes=3, max_iter=10000, alpha=1.0/1)
text_mlp_model = MLPClassifier(hidden_layer_sizes=5, max_iter=10000, alpha=1.0/5)
votes_mlp_model = MLPClassifier(hidden_layer_sizes=5, max_iter=10000, alpha=1.0/5)
dummy = DummyClassifier(strategy='most_frequent')
combined_fpr, combined_tpr, combined_auc = multiclass_roc_curves(combined_X, y, combined_mlp_model)
text_fpr, text_tpr, text_auc = multiclass_roc_curves(text_X, y, text_mlp_model)
votes_fpr, votes_tpr, votes_auc = multiclass_roc_curves(others, y, votes_mlp_model)
dummy_fpr, dummy_tpr, dummy_auc = multiclass_roc_curves(combined_X, y, dummy)

plt.plot(combined_fpr, combined_tpr, label="Combine text and vote features. AUC = " + str(round(combined_auc, 2)), color="navy")
plt.plot(text_fpr, text_tpr, label="Just text features. AUC = " + str(round(text_auc, 2)), color="yellow")
plt.plot(votes_fpr, votes_tpr, label="Just vote features. AUC = " + str(round(votes_auc, 2)), color="green")
plt.plot(dummy_fpr, dummy_tpr, label="Baseline classifier. AUC = " + str(round(dummy_auc, 2)), color="red")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Macro ROC of feature types")
plt.legend(loc="lower right")
plt.show()
