"""
Simple simulation code to reproduce figures on hight variance of
the test set
"""

import pandas
import numpy as np
from scipy import ndimage

from joblib import Parallel, delayed, Memory
from sklearn.model_selection import (GroupShuffleSplit, LeaveOneOut,
        cross_val_score)
from sklearn.metrics import accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

###############################################################################
# Code to run the experiments

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
# The perfect predictor

class PerfectEstimator(BaseEstimator, ClassifierMixin):

    def fit(self, X, y):
        return self

    def predict(self, X):
        return (X.mean(axis=1) > 0).astype(np.int)


###############################################################################
# Code to run the experiments

def sample_test_sets(train_size=200, noise_corr=2, dim=3, sep=.5,
                             random_state=0):
    """ Runs an experiments and returns the corresponding lines in
        the results dataframe.
    """
    X, y = mk_data(n_samples=train_size,
                   separability=sep, random_state=random_state,
                   dim=dim)

    # Create 10 blocks of evenly-spaced labels for GroupShuffleSplit
    groups = np.arange(train_size) // (train_size // 10)

    scores = list()
    for name, cv in [('loo', LeaveOneOut()),
                     ('50 splits',
                      GroupShuffleSplit(n_splits=50, random_state=0))]:
        this_scores = cross_val_score(PerfectEstimator(), X, y,
                                      cv=cv, groups=groups)
        scores.append(dict(
            cv_name=name,
            validation_score=ACCURACY,
            train_size=train_size,
            dim=dim,
            noise_corr=noise_corr,
            sep=sep,
            score_error=(np.mean(this_scores) - ACCURACY),
            score_sem=(np.std(this_scores) / np.sqrt(len(this_scores))),
            ))

    return scores



###############################################################################
# Compute the true accuracy
SEPARABILITY = 3

X, y = mk_data(n_samples=100000,
            separability=SEPARABILITY, dim=300, noise_corr=0)
ACCURACY = accuracy_score(PerfectEstimator().predict(X), y)
print(ACCURACY)



###############################################################################
# Run the simulations

N_JOBS = -1
N_DRAWS = 1000
mem = Memory(cachedir='cache')


results = pandas.DataFrame(
    columns=['cv_name', 'validation_score', 'train_size', 'dim',
             'noise_corr', 'sep', 'score_error', 'score_sem'])


for train_size in (100, 300, 900):
    scores = Parallel(n_jobs=N_JOBS, verbose=10)(
                    delayed(mem.cache(sample_test_sets))(
                            train_size=train_size,
                            noise_corr=0, dim=300, sep=SEPARABILITY,
                            random_state=i)
                    for i in range(N_DRAWS))
    for line in scores:
        results = results.append(line)

results.to_csv('perfect_predictor_results.csv')


