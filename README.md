# Introduction to Machine Learning, Coursework 1
This project implements a nested cross-fold validation decision tree with branch pruning. The specification can be found [here](spec.pdf).
### [*Main Test File*](test.py)
This is the main file of our project and will provide an options menu for testing. \
Run using:
```bash
$ python3 test.py
```

#### [test.py](test.py)
- `train_and_test(dataset_path="wifi_db/clean_dataset.txt")` Tests the algorithm on a dataset and prints the evaluation metrics: 
  - Total confusion matrix from summing all the matrices from each fold.
  - Average accuracy.
  - Average precision and recall per class (room).
  - Average F1 score per class. This is done by calculating the F1 score per class for each fold, then finding the average across folds.
- `__name__ == "__main__"` This initialiser allows the user to train and test or visualise the tree on a clean or noisy dataset using both the base trained and the pruned versions of the tree.
- Note that visualising the tree will show the plot and also save it in `plot.png`.



### [*Training Folder*](training/)
-----
This folder contains the functions related to initially training and pruning the tree.

#### [build_tree.py](training/build_tree.py)
- `decision_tree_learn(training_dataset)` Trains a decision tree on the given dataset and returns a tuple of the root node and maximum depth. The training dataset is a numpy array where the last column contains the correct labels.

#### [prune_tree.py](training/prune_tree.py)
- `prune_tree(node, tree, training_dataset, validation_dataset)` Recursively goes through the nodes of the tree and if a node has two leaf children it will prune it based on the accuracy of the original and pruned version of the tree. It returns the final (pruned) tree, with its accuracy and confusion matrix, tested on the validation dataset.

### [*Evaluation Folder*](evaluation/)
-----
This folder contains the functions related to evaluation metrics and k_fold testing.

#### [eval_tree.py](evaluation/eval_tree.py)
- `classify(attributes, tree)` Finds the corresponding label of a list of attributes to a decision tree recursively. Knows when it's reached a leaf node by checking type.
- `evaluate(test_db, trained_tree)` Iterates through a dataset using `classify` for each row, while also updating the confusion matrix each iteration.
- `precision` and `recall` Both use a confusion matrix and also return the macro precision and recall.
- `f1_score_class(precision, recall)` Returns F1 score per class. Only this function for F1 score is used for efficiency so as to not calculate precision and recall metrics twice per class.

#### [k_folds.py](evaluation/k_folds.py)
- `indices_split_folds(fold_n, instances_n)` Shuffles indices and organises them into separate folds for each fold iteration.
- `validation_training_split(train_indices, k=10)` Splits the data into training and validation sets based on k-1 folds.
- `test_k_folds(dataset, k=10)` Tests the algorithm on a dataset for k-fold cross validation. Builds a tree for each fold, tests, and returns list of accuracies and confusion matrices for each fold.
- `test_k_folds_pruning(dataset, k=10)` Trains, prunes and tests k trees. Returns 2 arrays of dimension k with each accuracy and confusion matrix and the best pruned tree.

### [*Visualisation folder*](visualisation/)
-----
This folder contains the TreeVis class for tree visualisation, and a randomised testing program.

#### [tree_vis.py](visualisation/tree_vis.py)
- The TreeVis class is instantiated as `vis = TreeVis()`.
- To draw a tree starting from a root node `root`, call `vis.draw(root)`.
- The plot is blocking, and will suspend execution until closed.
- The plot is also saved in `plot.png`.

#### [tree_vis_testing.py](visualisation/tree_vis_testing.py)
- `basic_plot(root)` Plots a tree from the given root using the naive recursive algorithm (distance between adjacent nodes halves for each layer). For comparison purposes.
- `random_tree(node, p, max_depth)` Randomly generates a tree with a maximum depth starting from a given root node. `p` is the probability that at a node with no children, a left/right child will be created (so a higher value of p leads to more nodes).

`tree_vis_testing.py` can be run by calling:
```bash
$ python3 visualisation/tree_vis_testing.py
```

### [*Dataset Folder*](wifi_db/)
-----
This folder conatains both the clean and noisy datasets used for training and testing the decision trees built in this project.