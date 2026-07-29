import numpy as np # use numpy's random number generation

from mysklearn import myutils

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    rng = np.random.default_rng(random_state)
    
    if shuffle:
        indices = [i for i, _ in enumerate(X)]
        rng.shuffle(indices)
        X = [X[i] for i in indices]
        y = [y[i] for i in indices]

    if isinstance(test_size, float):
        index = int(np.ceil(test_size * len(X)))
    else:
        index = test_size

    X_train, X_test = X[:-index], X[-index:]
    y_train, y_test = y[:-index], y[-index:]


    return X_train, X_test, y_train, y_test # TODO: fix this

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    # Seed rng and handle the shuffle
    rng = np.random.default_rng(random_state)
    n_samples = len(X)
    if shuffle:
        indices = [i for i, _ in enumerate(X)]
        rng.shuffle(indices)
        X = [X[i] for i in indices]
    else:
        indices = [i for i in range(len(X))]

    # Generate the fold sizes
    fold_sizes = []
    for i in range(n_splits):
        if i < n_samples % n_splits:
            fold_sizes.append(n_samples // n_splits + 1)
        else:
            fold_sizes.append(n_samples // n_splits)
    
    iterator = 0
    folds = [] # List of the tuples
    for fold_size in fold_sizes:
        test_indxs = indices[iterator:iterator + fold_size]
        train_indxs = [i for i in indices if i not in test_indxs]
        #adder = tuple(train_indx, test_indx)
        folds.append((train_indxs, test_indxs))
        iterator += fold_size

    return folds # TODO: fix this

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    return [] # TODO: (BONUS) fix this

def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if n_samples is None:
        n_samples = len(X)
    
        
    rng = np.random.default_rng(random_state)

    sample_indices = []
    X_sample = []
    X_out_of_bag = []
    y_sample = []
    y_out_of_bag = []
    for i in range(n_samples):
        indx = rng.integers(0, len(X))
        sample_indices.append(indx)
    # sample n_samples
    for i, row in enumerate(X):
        if i in sample_indices:
            X_sample.append(row)
        else:
            X_out_of_bag.append(row)

    if y is None:
        y_sample = None
        y_out_of_bag = None
    else:
        for i, item in enumerate(y):
            if i in sample_indices:
                y_sample.append(item)
            else:
                y_out_of_bag.append(item)

    return X_sample, X_out_of_bag, y_sample, y_out_of_bag # TODO: fix this

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    n_samples = len(y_true)
    n_labels = len(labels)
    # initialize every index of matrix to 0 with shape of n_labels x n_labels for pred and tre classification
    matrix = [[0 for i in range(n_labels)] for j in range(n_labels)]
    for i in range(n_samples):
        true_indx = labels.index(y_true[i])
        test_indx = labels.index(y_pred[i])
        matrix[true_indx][test_indx] += 1

    return matrix # TODO: fix this

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    n_samples = len(y_true)
    correct = 0
    for i, item in enumerate(y_true):
        if item == y_pred[i]:
            correct += 1

    if normalize:
        return correct / n_samples
    
    return correct # TODO: fix this

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = []
        labels.append(y_true[0])
        for val in y_true:
            if val not in labels:
                labels.append(val)

    if pos_label is None:
        pos_label = labels[0]

    tp = 0
    fp = 0
    for yt, yp, in zip(y_true, y_pred):
        if yt == yp and yt == pos_label:
            tp += 1
        else:
            if yp == pos_label:
                fp += 1

    denom = tp + fp
    if denom == 0: return 0.0
    return tp / denom # TODO: fix this

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """

    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]
    
    tp = 0
    fn = 0
    for yt, yp in zip(y_true, y_pred):
        if yt == yp and yt == pos_label:
            tp += 1
        else:
            if yp != pos_label and yt == pos_label:
                fn += 1

    denom = tp + fn
    if denom == 0: return 0.0
    return tp / denom # TODO: fix this

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """

    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    numer = 2 * precision * recall
    denom = precision + recall
    if denom == 0: return 0.0
    return numer / denom # TODO: fix this
