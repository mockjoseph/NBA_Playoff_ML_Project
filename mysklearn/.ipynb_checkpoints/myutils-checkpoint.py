import numpy as np # use numpy's random number generation
from mysklearn import myevaluation
from tabulate import tabulate 
import matplotlib.pyplot as plt


def get_DOE_rating(mpg):
        if mpg >= 45:
            return 10
        elif mpg < 45 and mpg >= 37:
            return 9
        elif mpg < 37 and mpg >= 31:
            return 8
        elif mpg < 31 and mpg >= 27:
            return 7
        elif mpg < 27 and mpg >= 24:
            return 6
        elif mpg < 24 and mpg >= 20:
            return 5
        elif mpg < 20 and mpg >= 17:
            return 4
        elif mpg < 17 and mpg >= 15:
            return 3
        elif mpg == 14:
            return 2
        else:
            return 1

def get_instances(mypytable, key_columns):
    instances = []
    indices = [i for i, col in enumerate(mypytable.column_names) if col in key_columns]
    for i, row in enumerate(mypytable.data):
        instances.append([row[i] for i in indices])

    return instances
        
def get_DOE_rating_list(mpg):
    '''
    Takes in a list of mpg values and returns the parralel list for the DOE
    rating
    '''
    doe_ratings = []
    for val in mpg:
        if val >= 45:
            doe_ratings.append(10)
        elif val < 45 and val >= 37:
            doe_ratings.append(9)
        elif val < 37 and val >= 31:
            doe_ratings.append(8)
        elif val < 31 and val >= 27:
            doe_ratings.append(7)
        elif val < 27 and val >= 24:
            doe_ratings.append(6)
        elif val < 24 and val >= 20:
            doe_ratings.append(5)
        elif val < 20 and val >= 17:
            doe_ratings.append(4)
        elif val < 17 and val >= 15:
            doe_ratings.append(3)
        elif val == 14:
            doe_ratings.append(2)
        else:
            doe_ratings.append(1)
            
    return doe_ratings

def output_classified_results(X_test, y_test, predictions):
    for i, instance in enumerate(X_test):
        print("instance:",instance, "Class:", predictions[i], "Actual:", y_test[i])

    return


def random_subsample(X, y, k, mykNN, myDumm):
    X_train_samples = []
    y_train_samples = []
    X_test_samples = []
    y_test_samples = []
    knn_acc = 0
    dum_acc = 0
    for i in range(k):
        X_train, X_test, y_train, y_test = myevaluation.train_test_split(X, y, test_size=0.33, random_state=None, shuffle=False)
        mykNN.fit(X_train, y_train)
        knn_predictions = mykNN.predict(X_test)
        knn_actual = y_test
        knn_acc += myevaluation.accuracy_score(knn_actual, knn_predictions, normalize=True)
    

    print("===========================================")
    print("STEP 1: Predictive Accuracy")
    print("===========================================")
    print("Random Subsample (k=10, 2:1 Train/Test)")
    print("k Nearest Neighbors Classifier: accuracy =", knn_acc / k, "error rate =", 1 - (knn_acc / k))
    print("Dummy Classifier: accuracy =", dum_acc / k, "error rate= = ", 1 -(dum_acc / k))

    return X_train_samples, X_test_samples, y_train_samples, y_test_samples

def cross_val_predict(X, y, k, mykNN, myDum):
    folds = myevaluation.kfold_split(X, k)
    kNN_acc = 0
    kNN_precision = 0
    kNN_recall = 0
    kNN_F1 = 0
    
    dum_acc = 0
    dum_precision = 0
    dum_recall = 0
    dum_F1 = 0
    for fold in folds:
        X_train = [X[i] for i in fold[0]]
        y_train = [y[i] for i in fold[0]]
        mykNN.fit(X_train, y_train)
        myDum.fit(X_train, y_train)
        X_test = [X[i] for i in fold[1]]
        y_test = [y[i] for i in fold[1]]
        y_pred = mykNN.predict(X_test)
        y_true = y_test
        kNN_acc += myevaluation.accuracy_score(y_true, y_pred)
        kNN_precision += myevaluation.binary_precision_score(y_true, y_pred)
        kNN_recall += myevaluation.binary_recall_score(y_true, y_pred)
        kNN_F1 += myevaluation.binary_f1_score(y_true, y_pred)
        
        y_pred = myDum.predict(y_test)
        dum_acc += myevaluation.accuracy_score(y_true, y_pred)
        dum_precision += myevaluation.binary_precision_score(y_true, y_pred)
        dum_recall += myevaluation.binary_recall_score(y_true, y_pred)
        dum_F1 += myevaluation.binary_f1_score(y_true, y_pred)

    print("===================================================")
    print("Predictive Accuracy, Precision, Recall, F1 Measure")
    print("===================================================")
    print("10-Fold Cross Validation")
    print("k Nearest Neighbors Classifier: accuracy =", kNN_acc / k, "error rate =", 1 -(kNN_acc / k))
    print("k Nearest Neighbors Classifier: Precision =", kNN_precision / k, "error rate =", 1 -(kNN_precision / k))
    print("k Nearest Neighbors Classifier: Recall =", kNN_recall / k, "error rate =", 1 -(kNN_recall / k))
    print("k Nearest Neighbors Classifier: F1 =", kNN_F1 / k, "error rate =", 1 -(kNN_F1 / k))
    print("Dummy Classifier: accuracy =", dum_acc / k, "error rate =", 1 -(dum_acc / k))
    print("Dummy Classifier: Precision =", dum_precision / k, "error rate =", 1 -(dum_precision / k))
    print("Dummy Classifier: Recall =", dum_recall / k, "error rate =", 1 -(dum_recall / k))
    print("Dummy Classifier: F1 Measure =", dum_F1 / k, "error rate =", 1 -(dum_F1 / k))
    

def bootstrap_method(X, y, k, mykNN, myDum):
    knn_acc = 0
    dum_acc = 0
    for i in range(k):
        X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(X, y)
        mykNN.fit(X_train, y_train)
        myDum.fit(X_train, y_train)
        y_pred = mykNN.predict(X_test)
        y_true = y_test
        knn_acc += myevaluation.accuracy_score(y_true, y_pred)
        y_pred = myDum.predict(X_test)
        dum_acc += myevaluation.accuracy_score(y_true, y_pred)
        

    print("===========================================")
    print("STEP 3: Predictive Accuracy")
    print("===========================================")
    print("k=10 Bootstrap Method")
    print("k Nearest Neighbors Classifier: accuracy =", knn_acc / k, "error rate =", 1 -(knn_acc / k))
    print("Dummy Classifier: accuracy =", dum_acc / k, "error rate =", 1 -(dum_acc / k))

def matrix_method(X, y, k, mykNN, myDum):
    folds = myevaluation.kfold_split(X, k)
    for fold in folds:
        X_train = [X[i] for i in fold[0]]
        y_train = [y[i] for i in fold[0]]
        mykNN.fit(X_train, y_train)
        myDum.fit(X_train, y_train)
        X_test = [X[i] for i in fold[1]]
        y_test = [y[i] for i in fold[1]]
        y_pred = mykNN.predict(X_test)
        y_true = y_test
        knn_matrix = [[0 for i in range(10)]for i in range(10)]
        dum_matrix = [[0 for i in range(10)]for i in range(10)]
        for true, pred in zip(y_true, y_pred):
            knn_matrix[true - 1][pred - 1] += 1
            
        
        y_pred = myDum.predict(y_test)
        for true, pred in zip(y_true, y_pred):
            dum_matrix[true - 1][pred - 1] += 1

    knn_header = "naive matrix"
    knn_table = tabulate(knn_matrix, knn_header, tablefmt="grid")
    dum_header = "dum matrix"
    dum_table = tabulate(dum_matrix, dum_header, tablefmt="grid")
    print(dum_table)


def get_reg_season_games(mypytable):
    n_row, _ = mypytable.get_shape()
    if n_row > 81:
        drop_indices = list(range(82, n_row))
        mypytable.drop_rows(drop_indices)
    
    return mypytable

def get_first_n_games(mypytable, n):
    n_row, _ = mypytable.get_shape()
    drop_indices = list(range(n, n_row))
    mypytable.drop_rows(drop_indices)
    
    return mypytable

def get_scatter(x_col, y_col, table):
    plt.figure()
    xs = table.get_column(x_col)
    ys = table.get_column(y_col)
    plt.xlim(min(xs) * 0.8, max(xs) * 1.2)
    plt.ylim(min(ys) * 0.8, max(ys) * 1.2)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    
    plt.plot(xs, ys, "b.")

def create_boxplots(table):
    plt.figure()
    years = set(table.get_column("model year"))
    # Now have an unorded set of each year
    mpg_by_year = []
    year_index = table.column_names.index("model year")
    mpg_index = table.column_names.index("mpg")
    for year in years:
        add_mpgs = []
        for rows in table.data:
            if rows[year_index] == year:
                add_mpgs.append(rows[mpg_index])
        mpg_by_year.append(add_mpgs)

    plt.boxplot(mpg_by_year, years)

def assert_made_playoffs(instances):
    playoffs_made = []
    for val in instances:
        if val < 45:
            playoffs_made.append("no")
        else:
            playoffs_made.append("yes")

    return playoffs_made

def get_prediction_difference(actual, predicted):
    diffs = []
    for a, p in zip(actual, predicted):
        val = a - p
        diffs.append(abs(val))

    average = 0
    for val in diffs:
        average += val

    average = average / len(diffs)
    return average
    

# Decision tree helper functions


def get_full_train(X_train, y_train):
    new_train = [row.copy() for row in X_train]
    for row, val in zip(new_train, y_train):
        row.append(val)
    
    return new_train

def get_header(X_train):
    header = []
    for i, _ in enumerate(X_train[0]):
        header.append(f"att{i}")

    return header

def get_attribute_domains(header, X_train):
    attribute_domains = {}
    for i, val in enumerate(header):
        col_vals = list(sorted(set(row[i] for row in X_train)))
        if val not in attribute_domains:
            attribute_domains[val] = col_vals
    
    return attribute_domains

def sing_ent(instances):
    # Want to pass in a single value for given attribute
    if len(instances) == 0:
        return 0
    
    counts = {}

    for row in instances:
        if row[-1] not in counts:
            counts[row[-1]] = 0
        counts[row[-1]] += 1
    total = len(instances)
    pros = [count / total for count in counts.values()]
    ent = 0
    for prob in pros:
        ent += prob * np.log2(prob)
        #print(ent)
    #print(ent)
    return -ent

def calculate_attribute_entropy(instances, attribute, attribute_domain, header):
    att_ent = 0
    partitions = partition_instances(instances, attribute, attribute_domain, header)
    for partition in partitions.values():
        weight = len(partition) / len(instances)
        
        att_ent += weight * sing_ent(partition)

    return att_ent

def select_attribute(instances, attributes, attribute_domain, header):
    # TODO: implement the general Enew algorithm for attribute selection
    # for each available attribute
    #     for each value in the attribute's domain
    #          calculate the entropy for the value's partition
    #     calculate the weighted average for the parition entropies
    # select that attribute with the smallest Enew entropy
    # for now, select an attribute randomly
    best_ent = calculate_attribute_entropy(instances, attributes[0], attribute_domain, header)
    #print("test")
    best_attribute = attributes[0]
    for attribute in attributes:
        entrop = calculate_attribute_entropy(instances, attribute, attribute_domain, header)
        #print(entrop)
        if entrop < best_ent:
            #print(best_attribute)
            best_attribute = attribute

    
    #rand_index = np.random.randint(0, len(attributes))
    #print(best_attribute)
    return best_attribute

def partition_instances(instances, attribute, attribute_domains, header):
    # this is group by attribute domain (not values of attribute in instances)
    # Returns a dictionary: {attribute_value: [instances]}
    att_index = header.index(attribute)
    att_domain = attribute_domains[attribute]
    partitions = {}
    for att_value in att_domain: # "Junior" -> "Mid" -> "Senior"
        partitions[att_value] = []
        for instance in instances:
            if instance[att_index] == att_value:
                partitions[att_value].append(instance)

    return partitions

def all_same_class(instances):
    # get the class label of the first instance.
    first_class = instances[0][-1]
    for instance in instances:
        # if any label differs, return False immediately.
        if instance[-1] != first_class:
            return False
        
    # if the loop completes without finding differences, return True.
    return True 

def tdidt(current_instances, available_attributes, attribute_domain, header):
    
    #    Recursively building a decision tree using the TDIDT algorithm.

    #     1. Select the best attribute to split on and create an "Attribute" node.
    #     2. For each value of the selected attribute:
    #         a. Create a "Value" subtree.
    #         b. If all instances in this partition have the same class:
    #             - Append a "Leaf" node
    #         c. If there are no more attributes to select:
    #             - Append a "Leaf" node (handle clash w/majority vote leaf node)
    #         d. If the partition is empty:
    #             - Append a "Leaf" node (backtrack and replace attribute node with majority vote leaf node)
    #         e. Otherwise:
    #             - Recursively build another "Attribute" subtree for this partition
    #               and append it to the "Value" subtree.
    #     3. Append each "Value" subtree to the current "Attribute" node.
    #     4. Return the current tree (nested list structure).

    

    #print("available attributes:", available_attributes)
    
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes, attribute_domain, header)
    #print("splitting on:", split_attribute)
    available_attributes.remove(split_attribute) # can't split on this attribute again in this subtree

    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, attribute_domain, header)
    #print("partitions:", partitions)
    
    # for each partition, repeat unless one of the following occurs (base case)
    for att_value in sorted(partitions.keys()): # process in alphabetical order
        att_partition = partitions[att_value]
        #print("Attribute partition:", att_partition)
        value_subtree = ["Value", att_value]

        #    CASE 1: all class labels of the partition are the same
        # => make a leaf node
        if len(att_partition) > 0 and all_same_class(att_partition):
            #print("CASE 1")
            #print(att_partition)
            leaf = ["Leaf", att_partition[0][-1], len(att_partition), len(current_instances)]
            value_subtree.append(leaf)
            
            

        #    CASE 2: no more attributes to select (clash)
        # => handle clash w/majority vote leaf node
        elif len(att_partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            #labels = set(att_partition[-1])
            counts = {}
            for row in att_partition:
                if row[-1] not in counts:
                    counts[row[-1]] = 0
                counts[row[-1]] += 1

            majority = max(counts, key=counts.get)
            maxi = counts[majority]

            leaf = ["Leaf", majority, len(att_partition), len(current_instances)]
            value_subtree.append(leaf)
            


        #    CASE 3: no more instances to partition (empty partition)
        # => backtrack and replace attribute node with majority vote leaf node
        elif len(att_partition) == 0:
            #print("CASE 3")
            counts = {}
            for row in current_instances:
                if row[-1] not in counts:
                    counts[row[-1]] = 0
                counts[row[-1]] += 1
            majority = None
            maxi = 0
            for key, count in counts.items():
                if count > maxi:
                    maxi = count
                    majority = key
            leaf = ["Leaf", majority, maxi, len(current_instances)]
            value_subtree.append(leaf)

        else:
            # none of base cases were true, recurse!!
            subtree = tdidt(att_partition, available_attributes.copy(), attribute_domain, header)
            value_subtree.append(subtree)
            
        tree.append(value_subtree)
        # TODO: append subtree to value_subtree and value_subtree to tree appropriately
    return tree


def tdidt_predict(tree, instance, header):
    data_type = tree[0]

    # Base case: if this is a leaf, just return its class label
    if data_type == "Leaf":
        label = tree[1]
        return label
    
    # Recursive case:if we are here, this is an Attribute node
    attribute_name = tree[1]
    attribute_index = header.index(attribute_name)
    instance_value = instance[attribute_index]

    # Look for the matching value node
    for values in tree[2:]:
        value = values[1]
        subtree = values[2]
        
        if instance_value == value:
            return tdidt_predict(subtree, instance, header)
        
def bin_nba_data(data):
    binned_data = []
    num_cols = len(data[0])
    cols = [list(col) for col in zip(*data)]
    # Now have list of lists cols
    cutoffs = []
    num_vals = len(cols[0])
    for col in cols:
        cutoff = []
        sorted_col = sorted(col)
        # Value at index 1/3 
        pct_33 = sorted_col[int(num_vals * 1/3)]
        # Value at index 2/3
        pct_66 = sorted_col[int(num_vals * 2/3)]
        cutoff.append(pct_33)
        cutoff.append(pct_66)
        cutoffs.append(cutoff)
    
    for cutoff, col in zip(cutoffs, cols):
        for i, val in enumerate(col):
            if val < cutoff[0]:
                col[i] = "bad"
            elif val > cutoff[1]:
                col[i] = "good"
            else:
                col[i] = "average"
    
    binned_data = [list(row) for row in zip(*cols)]

    return binned_data


def nba_class_performance_view(actual, predicted):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for a, p in zip(actual, predicted):
        if a == p and a == 'yes':
            tp += 1
        elif a == p and a == 'no':
            tn += 1
        elif a != p and a == 'yes':
            fn += 1
        else:
            fp += 1

   
    
    headers = ['', 'Yes', 'No']
    data = [['Yes', tp, fn],
            ['No', fn, tn]]
    print(tabulate(data, headers, tablefmt='grid'))

 