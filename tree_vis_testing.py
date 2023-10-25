import random
from tree_vis import *

# root = {'split feature' : 0, 'split value' : 5,
#         'left' : {
#             'split feature' : 1, 'split value' : 10, 'left' : 0, 'right' : {
#                 'split feature' : 2, 'split value' : 50, 'left' : {
#                     'split feature' : 0, 'split value' : 20, 'left' : 0, 'right' : 0, 'depth' : 3
#                 }, 'right' : {
#                     'split feature' : 3, 'split value' : 100, 'left' : {
#                         'split feature' : 4, 'split value' : 300, 'left' : 0, 'right' : 1, 'depth' : 4
#                     }, 'right' : 8, 'depth' : 3
#                 }, 'depth' : 2
#             }, 'depth' : 1
#         },
#         'right' : {
#             'split feature' : 1, 'split value' : 20, 'left' : {
#                 'split feature' : 2, 'split value' : 200, 'left' : {
#                     'split feature' : 3, 'split value' : 300, 'left' : 8, 'right' : 9, 'depth' : 3
#                 }, 'right' : 10, 'depth' : 2
#             }, 'right' : 3, 'depth' : 1
#         },
#         'depth' : 0}

# Generate a random tree.
# node: the root node to start the tree from.
# p: the probability that a left/right child will be created for a current leaf node.
# max_depth: the maximum depth the tree is allowed to reach.
def random_tree(node, p, max_depth):
    if node['depth'] == max_depth:
        return
    if random.random() < p:
        node['left'] = {'split feature' : 1, 'split value' : 10, 'left' : 0, 'right' : 0, 'depth' : node['depth'] + 1}
        random_tree(node['left'], p, max_depth)
    if random.random() < p:
        node['right'] = {'split feature' : 1, 'split value' : 10, 'left' : 0, 'right' : 0, 'depth' : node['depth'] + 1}
        random_tree(node['right'], p, max_depth)

root = {'split feature' : 1, 'split value' : 10, 'left' : 0, 'right' : 0, 'depth' : 0}
random_tree(root, 0.6, 10)

# See basic plot first.
plt.figure()
plt.axis('off')
basic_plot(root)

# Comare to Reingold-Tilford algorithm.
vis_tree(root)