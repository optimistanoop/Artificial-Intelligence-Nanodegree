import math
import statistics
import warnings

import numpy as np
from hmmlearn.hmm import GaussianHMM
from sklearn.model_selection import KFold
from asl_utils import combine_sequences


class ModelSelector(object):
    '''
    base class for model selection (strategy design pattern)
    '''

    def __init__(self, all_word_sequences: dict, all_word_Xlengths: dict, this_word: str,
                 n_constant=3,
                 min_n_components=2, max_n_components=10,
                 random_state=14, verbose=False):
        self.words = all_word_sequences
        self.hwords = all_word_Xlengths
        self.sequences = all_word_sequences[this_word]
        self.X, self.lengths = all_word_Xlengths[this_word]
        self.this_word = this_word
        self.n_constant = n_constant
        self.min_n_components = min_n_components
        self.max_n_components = max_n_components
        self.random_state = random_state
        self.verbose = verbose

    def select(self):
        raise NotImplementedError

    def base_model(self, num_states):
        # with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # warnings.filterwarnings("ignore", category=RuntimeWarning)
        try:
            hmm_model = GaussianHMM(n_components=num_states, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(self.X, self.lengths)
            if self.verbose:
                print("model created for {} with {} states".format(self.this_word, num_states))
            return hmm_model
        except:
            if self.verbose:
                print("failure on {} with {} states".format(self.this_word, num_states))
            return None


class SelectorConstant(ModelSelector):
    """ select the model with value self.n_constant

    """

    def select(self):
        """ select based on n_constant value

        :return: GaussianHMM object
        """
        best_num_components = self.n_constant
        return self.base_model(best_num_components)


class SelectorBIC(ModelSelector):
    """ select the model with the lowest Bayesian Information Criterion(BIC) score

    http://www2.imm.dtu.dk/courses/02433/doc/ch6_slides.pdf
    Bayesian information criteria: BIC = -2 * logL + p * logN
    """

    def select(self):
        """ select the best model for self.this_word based on
        BIC score for n between self.min_n_components and self.max_n_components

        :return: GaussianHMM object
        """
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on BIC scores
        best_score = float("Inf")
        best_model = None

        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                base_hmm = self.base_model(n_components)
                logL = base_hmm.score(self.X, self.lengths)
                logN = np.log(len(self.X))
                n = n_components
                p = n * n + 2 * n * len(self.X[0]) - 1
                score = -2 * logL + p * logN

                if score < best_score:
                    best_score = score
                    best_model = base_hmm
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorDIC(ModelSelector):
    ''' select best model based on Discriminative Information Criterion

    Biem, Alain. "A model selection criterion for classification: Application to hmm topology optimization."
    Document Analysis and Recognition, 2003. Proceedings. Seventh International Conference on. IEEE, 2003.
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.58.6208&rep=rep1&type=pdf
    https://pdfs.semanticscholar.org/ed3d/7c4a5f607201f3848d4c02dd9ba17c791fc2.pdf
    DIC = log(P(X(i)) - 1/(M-1)SUM(log(P(X(all but i))
    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection based on DIC scores
        best_score = float("-Inf")
        best_model = None

        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                base_hmm = self.base_model(n_components)
                score = base_hmm
                scores = []
                for word in self.words:
                    if (word != self.this_word):
                        x_word, word_len = self.hwords[word]
                        scores.append(base_hmm.score(x_word, word_len))
                dic_score = score - np.mean(base_hmm)
                if dic_score > best_score:
                    best_score = dic_score
                    best_model = base_hmm
            return best_model
        except:
            return self.base_model(self.n_constant)


class SelectorCV(ModelSelector):
    ''' select best model based on average log Likelihood of cross-validation folds

    '''

    def select(self):
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # TODO implement model selection using CV
        best_score = float("-Inf")
        best_model = None
        splits = min(3, len(self.sequences))

        try:
            for n_components in range(self.min_n_components, self.max_n_components + 1):
                kf = KFold(n_splits=splits, random_state=self.random_state)
                scores = []
                for train_idx, test_idx in kf.split(self.sequences):
                    x_train, train_len = combine_sequences(train_idx, self.sequences)
                    base_hmm = GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=1000,
                                    random_state=self.random_state, verbose=False).fit(x_train, train_len)
                    x_test, test_len = combine_sequences(test_idx, self.sequences)
                    logL = base_hmm.score(x_test, test_len)
                    scores.append(logL)
                score = np.average(scores)
                if score > best_score:
                    best_score = score
                    best_model = base_hmm
            return best_model
        except:
            return self.base_model(self.n_constant)

