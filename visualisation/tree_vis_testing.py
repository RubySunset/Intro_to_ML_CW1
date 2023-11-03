import random
from tree_vis import *
import matplotlib.pyplot as plt


# We can test using this manually defined tree if desired.
root = {'split feature' : 0, 'split value' : 5,
        'left' : {
            'split feature' : 1, 'split value' : 10, 'left' : 0, 'right' : {
                'split feature' : 2, 'split value' : 50, 'left' : {
                    'split feature' : 0, 'split value' : 20, 'left' : 0, 'right' : 0, 'depth' : 3
                }, 'right' : {
                    'split feature' : 3, 'split value' : 100, 'left' : {
                        'split feature' : 4, 'split value' : 300, 'left' : 0, 'right' : 1, 'depth' : 4
                    }, 'right' : 8, 'depth' : 3
                }, 'depth' : 2
            }, 'depth' : 1
        },
        'right' : {
            'split feature' : 1, 'split value' : 20, 'left' : {
                'split feature' : 2, 'split value' : 200, 'left' : {
                    'split feature' : 3, 'split value' : 300, 'left' : 8, 'right' : 9, 'depth' : 3
                }, 'right' : 10, 'depth' : 2
            }, 'right' : 3, 'depth' : 1
        },
        'depth' : 0}


# Naive solution using preorder recursion. Looks bad with deep trees.
# node: root node of the tree.
def basic_plot(node, x=0, y=50):

    if not type(node) is dict: # Plot leaf node
        plt.text(x, y, str(round(node)), ha='center', va='center', size='small',
                 bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
    else:
        plt.text(x, y, str(node['split feature']) + ' < ' + str(node['split value']), ha='center', va='center', size='small',
                 bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
        y_next = y - 10
        x_left = x - 2**(1 - node['depth'])
        x_right = x + 2**(1 - node['depth'])
        plt.plot((x - 2**(1 - node['depth']), x, x + 2**(1 - node['depth'])), (y_next, y, y_next)) # Plot lines to children
        basic_plot(node['left'], x_left, y_next)
        basic_plot(node['right'], x_right, y_next)


# Generate a random tree.
# p: the probability that a left/right child will be created for a current leaf node.
# max_depth: the maximum depth the tree is allowed to reach.
def random_tree(p, max_depth, node=None):

    if node == None:
        node = {'split feature' : 1, 'split value' : 10, 'left' : 0., 'right' : 0., 'depth' : 0}
    if node['depth'] == max_depth:
        return
    if random.random() < p:
        node['left'] = {'split feature' : 1, 'split value' : 10, 'left' : 0., 'right' : 0., 'depth' : node['depth'] + 1}
        random_tree(p, max_depth, node['left'])
    if random.random() < p:
        node['right'] = {'split feature' : 1, 'split value' : 10, 'left' : 0., 'right' : 0., 'depth' : node['depth'] + 1}
        random_tree(p, max_depth, node['right'])
    return node

# Generate a random tree from the root.
root = random_tree(0.5, 10)

# See basic plot first.
plt.figure()
plt.axis('off')
basic_plot(root)

# Comare to Reingold-Tilford algorithm.
vis = TreeVis()
vis.draw(root)