import numpy as np

# Recursive function to classify and return label of a single row of attributes
def classify(attributes, tree): ### This function is provisional: modify to fit tree structure and check syntax
    if attributes[tree["split feature"]] < tree["split value"]:
        if type(tree["right"]).__name__ == 'int': ### This assumes that label stored as int in dictionary
            return tree["right"]
        else:
            return classify(attributes, tree["right"])
    else:
        if type(tree["left"]).__name__ == 'int':
            return tree["left"]
        else:
            return classify(attributes, tree["left"])

# Evaluates the 4x4 confusion matrix of the trained tree against test_db, and also returns accuracy
def evaluate(test_db, trained_tree): ### This function is still provisional: Correctness is not yet determined
    confusion = np.zeros((4, 4), dtype=np.int)
    for i in range(len(test_db)): ### For each row in test_db, predict and change confusion matrix
        prediction = classify(test_db[i, :6], trained_tree) # The attributes passed to classify function shouldn't include the label
        if (prediction == test_db[i, 7]):
            confusion[prediction, prediction] += 1
        else:
            confusion[test_db[i, 7], prediction] += 1 # rows of confusion matrix are actual classes and columns are predicted classes

    if np.sum(confusion) > 0:
        accuracy = np.trace(confusion)/np.sum(confusion)
    else:
        accuracy = 0

    return accuracy, confusion ### Does this need to be returned as tuple as done in the lab?

# Returns array with precision of each class and the macro-averaged precision
def precision(confusion): ### Provisional Function
    p = np.zeros(len(confusion))
    for i in range(len(confusion)):
        p[i] = confusion[i, i] / np.sum(confusion[:, i])

    if len(p) > 0:
        macro_p = np.mean(p)
    else:
        macro_p = 0
    
    ### Do we really need to calculate macro_p as well?
    ### Maybe we won't work out averages of all macro_p from all evaluations?

    return p, macro_p ### Does this need to be returned as tuple as in lab?

# Returns array with recalls of each class and the macro-averaged recall
def recall(confusion): ### Provisional Function
    r = np.zeros(len(confusion))
    for i in range(len(confusion)):
        r[i] = confusion[i, i] / np.sum(confusion[i, :])

    if len(r) > 0:
        macro_r = np.mean(r)
    else:
        macro_r = 0
    
    return r, macro_r