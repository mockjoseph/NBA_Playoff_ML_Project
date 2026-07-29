from mysklearn import myutils
from mysklearn import myevaluation
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor
import numpy as np
import graphviz

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        # Either one should theoretically work as it is just a fit for a linear regressor
        self.regressor = MySimpleLinearRegressor()
        self.regressor.fit(X_train, y_train)
        #pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        if self.regressor.slope is None or self.regressor.intercept is None:
            raise ValueError("Model is not fitted yet. Call fit() before predict().")
        
        

        return [self.regressor.predict(X_test)] # TODO: fix this

    def discretizer(self, y_train):
        for y in y_train:
            if y >= 100:
                self.discretizer


class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        # First will need to compute the prior probabilities
        # this will become a dictioary of probabilities with the key being the classifiers
        classif_counts = {}
        total_classif = len(y_train)
        for classif in y_train:
            if classif not in classif_counts:
                classif_counts[classif] = 1
            else:
                classif_counts[classif] += 1

        self.priors = {key: count / total_classif for key, count in classif_counts.items()}

        # Now we will calculate the conditionals
        # 
        # Initialize the dictionary structure I created in test file
        num_attributes = len(X_train[0])
        self.conditionals = {}
        for key in classif_counts:
            self.conditionals[key] = {}
            for i in range(num_attributes):
                self.conditionals[key][i] = {}
            
        n_features = len(X_train) # for computing conditionals

        total_conditionals = {}
        for key in classif_counts:
            # Given key (yes or something)
            key_rows = [x for x, y in zip(X_train, y_train) if y == key]
            for i in range(num_attributes):
                # Look at each column
                col = [row[i] for row in key_rows]
                total = len(col)
                # Consider each different value in column
                for val in sorted(set(col)):
                    # Calculate conditional
                    self.conditionals[key][i][val] = col.count(val) / total
                
        

        #print(self.priors)
        #print(self.conditionals)
        
        return self.priors, self.conditionals # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        # For each instance in X_test, need to calculate the probability of each attribute given each classification
        # Follow the algorithm to do so
        all_probs = {}
        for instance in X_test:
            key_probs = {}
            for key in self.conditionals:
                #Set to prir
                probability = self.priors[key]
                for attribute in self.conditionals[key]:  # Attribute relates to an index so should work
                    val = instance[attribute]
                    probability *= self.conditionals[key][attribute].get(val, 0)

                key_probs[key] = probability
            predict = max(key_probs, key=key_probs.get)
            y_predicted.append(predict)
        


        return y_predicted # TODO: fix this


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        for test_instance in X_test:
            dists = []
            for train_inst in self.X_train:
                #print("Example train instance:", self.X_train[0])
                #print("Example test instance:", X_test[0])
                #print("Types:", [type(x) for x in self.X_train[0]])
                euclidian_dist = np.sqrt(sum((test_instance[i] - train_inst[i])**2 for i in range(len(train_inst))))
                dists.append(euclidian_dist)
            distances.append(dists)
        
        k_nearest_distances = []
        k_nearest_indices = []
        for instance in distances:
            indexed = list(enumerate(instance))
            indexed.sort(key=lambda x: (x[1], x[0]))  # sort by distance
            k_nearest_indices.append([idx for idx, _ in indexed[:self.n_neighbors]])
            k_nearest_distances.append([dist for _, dist in indexed[:self.n_neighbors]])
        
        return k_nearest_distances, k_nearest_indices
        # Steps
        # Normalize Data put this into a utils function probs for better looking and working code
        '''
        minx = min(x[0] for x in self.X_train)
        maxx = max(x[0] for x in self.X_train)
        miny = min(x[1] for x in self.X_train) 
        maxy = max(x[1] for x in self.X_train)
        normalized_train = []
        normalized_test = []
        for instance in self.X_train:
            norm_instance = []
            x = (instance[0] - minx) / (maxx - minx)
            norm_instance.append(x)
            y = (instance[1] - miny) / (maxy - miny)
            norm_instance.append(y)
            normalized_train.append(norm_instance)
        for instance in X_test:
            norm_instance = []
            x = (instance[0] - minx) / (maxx - minx)
            norm_instance.append(x)
            y = (instance[1] - miny) / (maxy - miny)
            norm_instance.append(y)
            normalized_test.append(norm_instance)
        
        #for instance in X_train:
        #    instance[0] = (instance[0] - minx) / (maxx - minx)
        #    instance[1] = (instance[1] - miny) / (maxy - miny)
        '''
        # Calculate Distances how can we do this without rewriting original datasets
        two_d_dist = []
        for test_instance in X_test:
            distances = []
            for train_instance in self.X_train:
                #print(test_instance)
                #print(train_instance)
                dist = np.sqrt(((test_instance[0] - train_instance[0])**2) + ((test_instance[1] - train_instance[1])**2))
                #print(dist)
                distances.append(dist)
            
            two_d_dist.append(distances)
            
        # Get distances of Test
        # Sort by closest distances # TODO : redo and implement utility functions
        k_nearest_distances = []
        k_nearest_indices = []
        for instance in two_d_dist:
            indexed = list(enumerate(instance))
            indexed.sort(key=lambda x: (x[1], x[0]))  # sort by distance
            k_nearest_indices.append([idx for idx, _ in indexed[:self.n_neighbors]])
            k_nearest_distances.append([dist for _, dist in indexed[:self.n_neighbors]])
            '''
            nearest_distances = sorted(instance)
            k_nearest_distances.append(nearest_distances[:self.n_neighbors])
            nearest_indices = []
            for dist in nearest_distances[:self.n_neighbors]:
                nearest_indices.append(instance.index(dist))

            k_nearest_indices.append(nearest_indices)
            '''
        # Get indices with their distances that are closest
        return k_nearest_distances, k_nearest_indices # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        distances, indices = self.kneighbors(X_test)
        y_predicted = []
        # List of all the classifiers in
        for instances in indices:
            classifiers = [self.y_train[indexes] for indexes in instances]
            most_common_label = max(set(classifiers), key=classifiers.count)
            y_predicted.append(most_common_label)
        
        return y_predicted # TODO: fix this
    

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None
        self.n_features = None
        self.attribute_names = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # Build the set and the attribute domains:
        new_train = myutils.get_full_train(X_train, y_train)
        header = myutils.get_header(X_train)
        if self.attribute_names is None:
            self.attribute_names = header
        
        self.n_features = len(header)
        attribute_domains = myutils.get_attribute_domains(header, X_train)
        #print(header)
        #print(new_train)
        #print(attribute_domains)
        available_attributes = header.copy()
        self.tree = myutils.tdidt(new_train, available_attributes, attribute_domains, header)
    
        #print("tree:", self.tree)
        
        # Now can start fitting
        

        pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        header = myutils.get_header(X_test)
        y_predicted = []
        for instance in X_test:
            y_predicted.append(myutils.tdidt_predict(self.tree, instance, header))
       
        
        return y_predicted # TODO: fix this

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """

        # Im not doing this
        
        pass # TODO: fix this

    # BONUS method
    
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        if self.tree is None:
            print("Tree has not been fitted yet. Cannot visualize.")
            return
        
        dot = graphviz.Digraph(comment='Decision Tree', graph_attr={'rankdir': 'TB'})
        self.node_counter = 0
        
        display_attribute_names_map = {}
        if attribute_names is not None and len(attribute_names) == self.n_features:
            for i in range(self.n_features):
                display_attribute_names_map[self.attribute_names[i]] = attribute_names[i]
        else:
            for i in range(self.n_features):
                display_attribute_names_map[self.attribute_names[i]] = self.attribute_names[i]
        
        self._get_dot_code_recursive(self.tree, dot, None, None, display_attribute_names_map)
        
        dot.render(dot_fname, view=False, format='dot', cleanup=True)
        dot.render(pdf_fname, view=False, format='pdf', cleanup=True)
        print(f"Decision tree visualization saved to {pdf_fname}.pdf and {dot_fname}.dot")

    def _get_dot_code_recursive(self, node, dot, parent_node_id, edge_label, display_attribute_names_map):
        current_node_id = str(self.node_counter)
        self.node_counter += 1
        if node[0] == "Leaf":
            label = f"Class: {repr(node[1])}\n({node[2]}/{node[3]})"
            dot.node(current_node_id, label, shape="ellipse", style="filled", fillcolor="lightgreen")
        else:
            internal_att_name = node[1]
            display_att_name = display_attribute_names_map[internal_att_name]
            label = f"Split on: {display_att_name}"
            dot.node(current_node_id, label, shape="box", style="filled", fillcolor="lightblue")

        if parent_node_id is not None:
            dot.edge(parent_node_id, current_node_id, label=str(edge_label))
            
        if node[0] == "Attribute":
            for i in range(2, len(node)):
                value_branch = node[i]
                value = value_branch[1]
                subtree = value_branch[2]
                self._get_dot_code_recursive(subtree, dot, current_node_id, repr(value), display_attribute_names_map)
        
        return current_node_id
        

class MyRandomForestsClassifier:
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.forest = []

    def fit(self, X, y, n_trees=3):

        for i in range(n_trees):
            tree = MyDecisionTreeClassifier()
            X_train, X_test, y_train, y_test = myevaluation.bootstrap_sample(X, y)

            tree.fit(X_train, y_train)
            self.forest.append(tree)
            
        return self.forest
        

    def predict(self, X_test):
        '''
        X_test = list of list object (list of the instances to predict)
        '''
        predicted = []
        for instance in X_test:
            all_preds = []
            for tree in self.forest:
                test_instance = []
                test_instance.append(instance)
                preds = tree.predict(test_instance)
                all_preds.append(preds[0])
                #all_preds.append(tree.predict(test_instance))


            vote = max(set(all_preds), key=all_preds.count)
            predicted.append(vote)
            



        
        return predicted # TODO: fix this
        




class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        counts = {}
        for classifier in y_train:
            if classifier in counts:
                counts[classifier] += 1
            else:
                counts[classifier] = 1

        max_count = 0
        for classifer, count in counts.items():
            if count > max_count:
                max_count = count
                self.most_common_label = classifier

        #pass # TODO: fix this

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for instances in X_test:
            y_predicted.append(self.most_common_label)

        return y_predicted # TODO: fix this
    
