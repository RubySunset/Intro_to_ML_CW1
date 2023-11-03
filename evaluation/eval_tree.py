import numpy as np


# Recursive function to classify and return predicted label of a single row of attributes
def classify(attributes, tree):

    if attributes[tree["split feature"]] > tree["split value"]:
        # The label is stored as float64 in the dictionary.
        if type(tree["right"]).__name__ == 'float64':
            return tree["right"]
        else:
            return classify(attributes, tree["right"])
    else:
        if type(tree["left"]).__name__ == 'float64':
            return tree["left"]
        else:
            return classify(attributes, tree["left"])


# Evaluates the 4x4 confusion matrix of the trained tree against test_db, and also returns accuracy
def evaluate(test_db, trained_tree):
    
    confusion = np.zeros((4, 4), dtype=np.int32)
    # For each row in test_db, predict and change confusion matrix
    for i in range(len(test_db)):
        # The attributes passed to classify function shouldn't include the label
        prediction = classify(test_db[i, :-1], trained_tree)
        # Rows of confusion matrix are actual classes and columns are predicted classes
        if (prediction == test_db[i, -1]):
            confusion[int(prediction)-1, int(prediction)-1] += 1
        else: 
            confusion[int(test_db[i, -1])-1, int(prediction)-1] += 1

    if np.sum(confusion) > 0:
        accuracy = np.trace(confusion)/np.sum(confusion)
    else:
        accuracy = 0

    return accuracy, confusion


# Returns array with precision of each class and the macro-averaged precision
def precision(confusion):

    p = np.zeros(len(confusion))
    for i in range(len(confusion)):
        p[i] = confusion[i, i] / np.sum(confusion[:, i])

    if len(p) > 0:
        macro_p = np.mean(p)
    else:
        macro_p = 0

    return p, macro_p

# Returns array with recalls of each class and the macro-averaged recall
def recall(confusion):

    r = np.zeros(len(confusion))
    for i in range(len(confusion)):
        r[i] = confusion[i, i] / np.sum(confusion[i, :])

    if len(r) > 0:
        macro_r = np.mean(r)
    else:
        macro_r = 0
    
    return r, macro_r

# Returns array with F1 scores of each class and macro-averaged F1 score.
# We define macro-averaged F1 score as being the mean of F1 scores across each class.
def f1_score_confusion(confusion):

    f = np.zeros(len(confusion))
    precisions = precision(confusion)[0]
    recalls = recall(confusion)[0]
    for i in range(len(confusion)):
        f[i] = (2*precisions[i]*recalls[i])/(precisions[i]+recalls[i])
    
    if len(f)>0:
        macro_f = np.mean(f)
    else:
        macro_f = 0
    
    return f, macro_f

# Returns the F1 score for each class
def f1_score_class(precision, recall):
    return ((2*precision*recall)/(precision+recall))