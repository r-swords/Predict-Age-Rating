from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.metrics import get_scorer_names, auc, roc_curve, roc_auc_score, classification_report, confusion_matrix
from sklearn.dummy import DummyClassifier
import matplotlib.pyplot as plt
import nltk
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import label_binarize
from sklearn.svm import LinearSVC

print(get_scorer_names())

def cv_plot(cv_range, mean_error, std_error, hyper_param, title):
    plt.title(title)
    plt.errorbar(cv_range, mean_error, yerr=std_error, linewidth=2)
    plt.xlabel(hyper_param)
    plt.ylabel('F1 macro')
    plt.show()

def confusion_matrix_summary(model, X, y):
    Xtr, Xte, Ytr, Yte = train_test_split(X, y, test_size=0.2)
    model.fit(Xtr, Ytr)
    print()
    print('test')
    print()
    print(classification_report(Yte, model.predict(Xte)))
    print(confusion_matrix(Yte, model.predict(Xte)))
    print()
    print('train')
    print()
    print(classification_report(Ytr, model.predict(Xtr)))
    print(confusion_matrix(Ytr, model.predict(Xtr)))

df = pd.read_csv("dataset.tsv", sep="\t")
text = df.content.astype("U")
others = df.iloc[:, 2:22]
y = df.rating

kNN_model = KNeighborsClassifier(n_neighbors=6, metric='manhattan')

svm_vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=50, max_df=300,
                             ngram_range=(1, 1))
svm_X = svm_vectoriser.fit_transform(text)
svm_X = pd.concat([pd.DataFrame(svm_X.toarray()), others], axis=1)
SVM_model = LinearSVC(C=1)

mlp_vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=10, max_df=200,
                                 ngram_range=(1, 4))
mlp_X = mlp_vectoriser.fit_transform(text)
mlp_X = pd.concat([pd.DataFrame(mlp_X.toarray()), others], axis=1)
mlp_model = MLPClassifier(hidden_layer_sizes=3, max_iter=10000, alpha=1.0/1)

dummy = DummyClassifier(strategy='most_frequent')

models = [[kNN_model, others], [SVM_model, svm_X], [mlp_model, mlp_X], [dummy, text]]
mean_error = []
std_error = []
for i in models:
    X = i[1]

    model =i[0]
    # conduct cross validation
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_macro')
    # record results
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
cv_plot(['kNN', 'SVM', 'MLP', 'Dummy'], mean_error, std_error, 'metric', 'Performance of all models')

confusion_matrix_summary(kNN_model, others, y)
confusion_matrix_summary(SVM_model, svm_X, y)
confusion_matrix_summary(mlp_model, mlp_X, y)
confusion_matrix_summary(dummy, text, y)