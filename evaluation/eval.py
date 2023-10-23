#Recursive function to classify and return label of a single row of attributes.
def classify(attributes, tree): ### This function is heavily reliant on tree structure, modify code to fit tree structure
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