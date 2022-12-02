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


def multiclass_roc_curves(X, Y, classifier):
    classes = ['G', 'PG', 'PG-13', 'R', 'NC-17']
    Y = label_binarize(Y, classes=classes)
    N = len(classes)
    Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.3)
    Ysc = np.array(classifier.fit(Xtr, Ytr))[
        :, :, 1].transpose()
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


vectoriser = TfidfVectorizer(stop_words=nltk.corpus.stopwords.words('english'), min_df=0.1, max_df=100,
                             ngram_range=(1, 2))


X = vectoriser.fit_transform(text)
X = pd.concat([pd.DataFrame(X.toarray()), others], axis=1)
scale = StandardScaler(with_mean=False)
scale.fit_transform(X)

Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2)  # for normal x
# Xtrain, Xtest, ytrain, ytest = train_test_split(
#     text, y, test_size=0.2)  # for text only x
# Xtrain, Xtest, ytrain, ytest = train_test_split(
#     others, y, test_size=0.2)  # for numbers only x

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
                    alpha=1.0/5,
                    learning_rate_init=0.01, max_iter=10000)

# Fit data onto the model
Xtrain = pd.get_dummies(Xtrain)
clf.fit(Xtrain, ytrain)


# Make prediction on test dataset
ypred = clf.predict(Xtest)

# Import accuracy score

# Calcuate accuracy
print(accuracy_score(ytest, ypred))


# iris = datasets.load_iris()
# X, y = iris.data, iris.target

y = label_binarize(y, classes=["R", "PG", "PG-13", "NC-17", "G"])
n_classes = 5

# shuffle and split training and test sets
X_train, X_test, y_train, y_test =\
    train_test_split(X, y, test_size=0.2, random_state=0)

#############################################################################################################################


#############################################################################################################################

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


# First aggregate all false positive rates
all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
# Plot all ROC curves
plt.figure()
plt.plot(
    fpr["micro"],
    tpr["micro"],
    label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
    color="deeppink",
    linestyle=":",
    linewidth=4,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
    color="navy",
    linestyle=":",
    linewidth=4,
)

colors = cycle(["aqua", "darkorange", "cornflowerblue"])
for i, color in zip(range(n_classes), colors):
    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        # lw=lw,
        label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
    )

# plt.plot([0, 1], [0, 1], "k--", lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC for feature types")
plt.plot([0, 1], [0, 1], 'k--')
plt.legend(loc="lower right")
plt.show()

# defunct for now....
# fpr['micro'], tpr['micro'], _ = roc_curve(y_train.ravel(), y_score.ravel())
# # try combining them

# all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))
# mean_tpr = np.zeros_like(all_fpr)

# for i in range(n_classes):
#     mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

#     # Finally average it and compute AUC
# mean_tpr /= n_classes

# scoreC = roc_auc_score(ytrain, y_score, average='macro', multi_class='ovr')

# plt.plot(all_fpr, mean_tpr, label="Combine text and vote features. AUC = " +
#          str(round(scoreC, 2)), color="navy")


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


print("We got to the plot section for F1")
plt.figure()
hidden_layer_range = [1, 2, 3, 5, 10, 25, 50, 75, 100]
mean_error = []
std_error = []
for n in hidden_layer_range:
    clf3 = MLPClassifier(hidden_layer_sizes=(n))
    print("hidden layer size %d\n" % n)
    scores = cross_val_score(clf3, X, y, cv=5, scoring='f1_weighted')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())

plt.errorbar(hidden_layer_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel('#hidden layer nodes')
plt.ylabel('F1')
plt.title("Cross validation scores for hidden layer size's")
plt.show()


mean_error = []
std_error = []
C_range = [1, 5, 10, 100, 1000]
for Ci in C_range:
    print("C %d\n" % Ci)
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(hidden_layer_sizes=(5), alpha=1.0/Ci)
    from sklearn.model_selection import cross_val_score
    scores = cross_val_score(model, X, y, cv=5, scoring='f1_weighted')
    mean_error.append(np.array(scores).mean())
    std_error.append(np.array(scores).std())
plt.errorbar(C_range, mean_error, yerr=std_error, linewidth=3)
plt.xlabel('C')
plt.ylabel('F1')
plt.title("Cross validation scores for hidden layer C range")
plt.show()
