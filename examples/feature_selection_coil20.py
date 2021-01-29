import scipy.io
import numpy as np

from sklearn import svm
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score

from skfeature.function.similarity_based import fisher_score

mat = scipy.io.loadmat("../data/COIL20.mat")
X=mat['X']

y = mat['Y'][:, 0]

n_samples, n_features = np.shape(X)
print(n_samples, n_features)

# label is always represented by a 1D vector, in the shape of (n_labels,)
n_labels = np.shape(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 40)

score = fisher_score.fisher_score(X_train, y_train)
print(score)

idx = fisher_score.feature_ranking(score)
print(idx)



num_fea = 5

selected_features_train = X_train[:, idx[0:num_fea]]

selected_features_test = X_test[:, idx[0:num_fea]]

clf = svm.LinearSVC()
clf.fit(selected_features_train, y_train)

y_predict = clf.predict(selected_features_test)
print(y_predict)

acc = accuracy_score(y_test, y_predict)

print(acc)
