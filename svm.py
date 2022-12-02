import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC, SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import nltk
from sklearn.preprocessing import StandardScaler, label_binarize

plt.rcParams['figure.constrained_layout.use'] = True


def multiclass_roc_curves(X, Y, classifier):

    # for finding model coeficients
    # model = classifier
    # Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)
    # model.fit(Xtrain, Ytrain)
    # print(model.coef_)
    # print(model.classes_)

    classes = ['G', 'PG', 'PG-13', 'R', 'NC-17']
    Y = label_binarize(Y, classes=classes)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3)

    clf = OneVsRestClassifier(classifier)
    try:
        # LinearSVC
        Ysc = np.array(clf.fit(Xtr, Ytr).decision_function(Xte))
    except:
        # Dummy classifier
        Ysc = np.array(clf.fit(Xtr, Ytr).predict_proba(Xte))
    N = len(classes)


    fpr, tpr = {}, {}
    for i in range(N):
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


def cv_plot(cv_range, mean_error, std_error, hyper_param, log):
    plt.errorbar(cv_range, mean_error, yerr=std_error, linewidth=3)
    if log:
        plt.xscale("log")
    plt.xlabel(hyper_param)
    plt.ylabel('F1 weighted')
    plt.show()


def min_df_cv(min_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=min_df_val)


def max_df_cv(max_df_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=20, max_df=max_df_val)


def ngram_cv(ngram_val):
    return TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=20, max_df=200,
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
        #X = pd.concat([pd.DataFrame(), votes], axis=1)
        #X = pd.concat([pd.DataFrame(X.toarray()), votes], axis=1)

        scale = StandardScaler(with_mean=False)
        scale.fit_transform(X)
        model = LinearSVC(C=1)
        # conduct cross validation
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
        # record results
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())
    cv_plot(cv_range, mean_error, std_error, hyper_param, False)


def penalty_cv(text, votes):
    vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=20, max_df=200,
                                 ngram_range=(1, 1))
    X = vectoriser.fit_transform(text)
    #X = pd.concat([pd.DataFrame(), votes], axis=1)
    #X = pd.concat([pd.DataFrame(X.toarray()), votes], axis=1)
    c_range = [ 0.00000001, 0.0000001, 0.000001, 0.00001, 0.0001, 0.001, 0.01,
               0.1, 1, 10, 100, 1000, 10000, 100000, 1000000, 10000000, 100000000, 1000000000, 10000000000,
               100000000000]
    mean_error = []
    std_error = []

    for c in c_range:
        model = LinearSVC(C = c)
        scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
        mean_error.append(np.array(scores).mean())
        std_error.append(np.array(scores).std())

    cv_plot(c_range, mean_error, std_error, 'penalty C', True)


df = pd.read_csv("dataset.tsv", sep="\t")
text = df.content.astype("U")
others = df.iloc[:, 2:22]
print(others)
y = df.rating

vectoriser_cv('min df', text, [1, 5, 10, 20, 30, 40, 50], others)
vectoriser_cv('max df', text, [50, 100, 200, 300, 400, 500, 1000, 1500, 2000], others)
vectoriser_cv('ngram', text, [1, 2, 3, 4, 5, 6, 7, 8, 9], others)
penalty_cv(text, others)

combined_vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=50, max_df=300,
                             ngram_range=(1, 1))
combined_X = combined_vectoriser.fit_transform(text)
combined_X = pd.concat([pd.DataFrame(combined_X.toarray()), others], axis=1)

text_vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=20, max_df=200,
                                 ngram_range=(1, 1))


text_X = text_vectoriser.fit_transform(text)

combined_SVM_model = LinearSVC(C=1)
text_SVM_model = LinearSVC(C=1)
votes_SVM_model = LinearSVC(C=1)
dummy = DummyClassifier(strategy='most_frequent')
multiclass_roc_curves(combined_X, y, combined_SVM_model)
combined_fpr, combined_tpr, combined_auc = multiclass_roc_curves(combined_X, y, combined_SVM_model)
text_fpr, text_tpr, text_auc = multiclass_roc_curves(text_X, y, text_SVM_model)
votes_fpr, votes_tpr, votes_auc = multiclass_roc_curves(others, y, votes_SVM_model)
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
