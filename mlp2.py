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
nltk.download('stopwords')


df = pd.read_csv("dataset.tsv", sep="\t")
text = df.content.astype("U")
others = df.iloc[:, 2:22]
y = df.rating


vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.1, max_df=100,
                             ngram_range=(1, 2))


X = vectoriser.fit_transform(text)
X = pd.concat([pd.DataFrame(X.toarray()), others], axis=1)
scale = StandardScaler(with_mean=False)
scale.fit_transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)

print(Xtest.head())
print(Xtrain.head())
print(ytest.head())
print(ytest.head())

unique_words = set(y)             # == set(['a', 'b', 'c'])
print(len(unique_words))
print(unique_words)  # == 3

# Create model object
clf = MLPClassifier(hidden_layer_sizes=(6, 5),
                    random_state=5,
                    verbose=True,
                    learning_rate_init=0.01, max_iter=10000)

# Fit data onto the model
clf.fit(Xtrain, ytrain)


# Make prediction on test dataset
ypred = clf.predict(Xtest)

# Import accuracy score

# Calcuate accuracy
print(accuracy_score(ytest, ypred))


# model = MLPClassifier(hidden_layer_sizes=(5), alpha=1.0/5).fit(Xtrain, ytrain)
# preds = model.predict(Xtest)
# print(confusion_matrix(ytest, preds))
# dummy = DummyClassifier(strategy="most_frequent").fit(Xtrain, ytrain)
# ydummy = dummy.predict(Xtest)
# print(confusion_matrix(ytest, ydummy))
# preds = model.predict_proba(Xtest)
# print(model.classes_)
# fpr, tpr, _ = roc_curve(ytest, preds[:, 1], pos_label=1)
# plt.plot(fpr, tpr)
# model = LogisticRegression(C=10000).fit(Xtrain, ytrain)
# fpr, tpr, _ = roc_curve(ytest, model.decision_function(Xtest))
# plt.plot(fpr, tpr, color='orange')
# plt.legend(['MLP', 'Logistic Regression'])
# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.plot([0, 1], [0, 1], color='green', linestyle='−−')
# plt.show()


# iris = datasets.load_iris()
# X, y = iris.data, iris.target

y = label_binarize(y, classes=["R", "PG", "from", "NC", "PG-13", "NC-17", "G"])
n_classes = 7

# shuffle and split training and test sets
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.33, random_state=0)

# classifier
# clf = OneVsRestClassifier(LinearSVC(random_state=0))
y_score = clf.fit(X_train, y_train).predict_proba(X_test)

# Compute ROC curve and ROC area for each class
fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# Plot of a ROC curve for a specific class
for i in range(n_classes):
    plt.figure()
    plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
