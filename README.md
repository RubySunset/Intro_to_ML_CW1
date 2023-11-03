# Intro_to_ML_CW1

Overview of [test_all.py](test_all.py):
- `train_and_test(dataset_path="wifi_db/clean_dataset.txt")` Tests the algorithm on a dataset and prints the evaluation metrics: 
  - Total confusion matrix from summing all the matrices from each fold.
  - Average accuracy.
  - Average precision and recall per class (room).
  - Average F1 score per class. This is done by calculating the F1 score per class for each fold, then finding the average across folds.
- Initialiser (`_name==main_`): Allows the user to: train and test the tree on a clean dataset, noisy dataset, or custom dataset; or visualise the tree trained on the entire clean dataset.

`test_all.py` can be run by calling `python3 test_all.py`.

### [*Tree training*](build_tree.py)
-----
This file contains the functions needed to train a decision tree on a dataset.

- `decision_tree_learn(training_dataset)` trains a decision tree on the given dataset and returns a tuple of the root node and maximum depth. The training dataset is a numpy array where the last column contains the correct labels.

### [*Evaluation Folder*](evaluation/)
-----
This folder contains the functions related to evaluation metrics and k_fold testing.

Functions in [eval.py](evaluation/eval.py):
- `classify(attributes, tree)` Finds the corresponding label of a list of attributes to a decision tree recursively. Knows when it's reached a leaf node by checking type.
- `evaluate(test_db, trained_tree)` Iterates through a dataset using `classify` for each row, while also updating the confusion matrix each iteration.
- `precision` and `recall` Both use a confusion matrix and also return the macro precision and recall.
- `f1_score_class(precision, recall)` Returns F1 score per class. Only this function for F1 score is used for efficiency so as to not calculate precision and recall metrics twice per class.

Functions in [k_folds.py](evaluation/k_folds.py):
- `indices_split_folds(fold_n, instances_n)` Shuffles indices and organises them into separate folds for each fold iteration.
- `test_k_folds(dataset, k=10)` Tests the algorithm on a dataset for k-fold cross validation. Builds a tree for each fold, tests, and returns list of accuracies and confusion matrices for each fold.

### [*Tree visualisation folder*](tree_vis/)
-----
This folder contains the TreeVis class for tree visualisation, and a randomised testing program.

[tree_vis.py](tree_vis/tree_vis.py):
- The TreeVis class is instantiated as `vis = TreeVis()`.
- To draw a tree starting from a root node `root`, call `vis.draw(root)`.

[tree_vis_testing.py](tree_vis/tree_vis_testing.py):
- `basic_plot(root)` plots a tree from the given root using the naive recursive algorithm (distance between adjacent nodes halves for each layer). For comparison purposes.
- `random_tree(node, p, max_depth)` randomly generates a tree with a maximum depth starting from a given root node. `p` is the probability that at a node with no children, a left/right child will be created (so a higher value of p leads to more nodes).

`tree_vis_testing.py` can be run by calling `python3 tree_vis/tree_vis_testing.py`.
