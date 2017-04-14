"""
The cross-validation error varying the dimensionality.
"""

import pandas
import numpy as np
from scipy import ndimage

from joblib import Parallel, delayed, Memory
from sklearn.model_selection import (GroupShuffleSplit, LeaveOneOut,
        cross_val_score)
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC

def mk_data(n_samples=200, random_state=0, separability=1,
            noise_corr=2, dim=100):
    rng = np.random.RandomState(random_state)
    y = rng.random_integers(0, 1, size=n_samples)
    noise = rng.normal(size=(n_samples, dim))
    if not noise_corr is None and noise_corr > 0:
        noise = ndimage.gaussian_filter1d(noise, noise_corr, axis=0)
    noise = noise / noise.std(axis=0)
    # We need to decrease univariate separability as dimension increases
    centers = 4. / dim * np.ones((2, dim))
    centers[0] *= -1
    X = separability * centers[y] + noise
    return X, y


###############################################################################
# Code to run the cross-validations


def sample_and_cross_val_clf(train_size=200, noise_corr=2, dim=3, sep=.5,
                             random_state=0):
    """ Runs an experiments and returns the corresponding lines in
        the results dataframe.
    """
    clf = LinearSVC(penalty='l2', fit_intercept=True)

    n_samples = train_size + 10000
    X, y = mk_data(n_samples=n_samples,
                   separability=sep, random_state=random_state,
                   noise_corr=noise_corr, dim=dim)
    X_train = X[:train_size]
    y_train = y[:train_size]
    X_test = X[train_size:]
    y_test = y[train_size:]

    validation_score = accuracy_score(
                            y_test,
                            clf.fit(X_train, y_train).predict(X_test))

    # Create 10 blocks of evenly-spaced labels for GroupShuffleSplit
    groups = np.arange(train_size) // (train_size // 10)

    scores = list()
    for name, cv in [('loo', LeaveOneOut()),
                     ('50 splits',
                      GroupShuffleSplit(n_splits=50, random_state=0))]:
        this_scores = cross_val_score(clf, X_train, y_train, cv=cv,
                                      groups=groups)
        scores.append(dict(
            cv_name=name,
            validation_score=validation_score,
            train_size=train_size,
            dim=dim,
            noise_corr=noise_corr,
            sep=sep,
            score_error=(np.mean(this_scores) - validation_score),
            score_sem=(np.std(this_scores) / np.sqrt(len(this_scores))),
            ))

    return scores



###############################################################################
# Run the simulations

N_JOBS = -1
N_DRAWS = 1000
mem = Memory(cachedir='cache')


results = pandas.DataFrame(
    columns=['cv_name', 'validation_score', 'train_size', 'dim',
             'noise_corr', 'sep', 'score_error', 'score_sem'])


for dim, sep in [(300, 5.),
                 (10000, 60.),
                 (10, .5),
                 (1, .13),
                ]:
    if dim > 1000:
        # Avoid memory problems
        n_jobs = 20
    else:
        n_jobs = N_JOBS
    scores = Parallel(n_jobs=n_jobs, verbose=10)(
                    delayed(mem.cache(sample_and_cross_val_clf))(
                            train_size=100,
                            noise_corr=0, dim=dim, sep=sep,
                            random_state=i)
                    for i in range(N_DRAWS))
    for line in scores:
        results = results.append(line)

results.to_csv('dimensionality_results.csv')


