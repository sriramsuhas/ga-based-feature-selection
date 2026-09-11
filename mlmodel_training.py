import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, GroupKFold
from sklearn.metrics import balanced_accuracy_score

def _cv_splitter(y, groups=None, n_splits=5, seed=42):
    if groups is not None:
        return GroupKFold(n_splits=n_splits)
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

def train_rf_balanced(train_file, mask=None, groups=None, n_splits=5, seed=42):
    arr = np.load(train_file)
    X, y = arr['X'], arr['y']
    if mask is not None:
        X = X[:, mask.astype(bool)]
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        random_state=seed, n_jobs=-1
    )
    cv = _cv_splitter(y, groups=groups, n_splits=n_splits, seed=seed)
    if groups is not None:
        scores = []
        for tr_idx, te_idx in cv.split(X, y, groups=groups):
            clf.fit(X[tr_idx], y[tr_idx])
            pred = clf.predict(X[te_idx])
            scores.append(balanced_accuracy_score(y[te_idx], pred))
        return float(np.mean(scores))
    else:
        scores = cross_val_score(clf, X, y, cv=cv, scoring='balanced_accuracy', n_jobs=-1)
        return float(scores.mean())

def evaluate_rf_balanced_once(npz_train, npz_eval, mask=None, groups_train=None, seed=42):
    dtr = np.load(npz_train)
    Xtr, ytr = dtr['X'], dtr['y']
    dev = np.load(npz_eval)
    Xev, yev = dev['X'], dev['y']
    if mask is not None:
        Xtr = Xtr[:, mask.astype(bool)]
        Xev = Xev[:, mask.astype(bool)]
    clf = RandomForestClassifier(
        n_estimators=300, max_depth=None, min_samples_leaf=1,
        random_state=seed, n_jobs=-1
    )
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xev)
    return float(balanced_accuracy_score(yev, pred))
