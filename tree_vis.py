# Tree node dictionary fields:
# 'split feature'
# 'split value'
# 'right'
# 'left'
# 'depth'

import copy
import matplotlib.pyplot as plt

# Naive solution using preorder recursion. Looks bad with deep trees.
def basic_plot(node, x=0, y=50):
    if type(node) is int:
        plt.text(x, y, str(node))
    else:
        plt.text(x, y, str(node['split feature']) + ' < ' + str(node['split value']))
        y_next = y - 10
        x_left = x - 2**(1 - node['depth'])
        x_right = x + 2**(1 - node['depth'])
        plt.plot((x - 2**(1 - node['depth']), x, x + 2**(1 - node['depth'])), (y_next, y, y_next))
        basic_plot(node['left'], x_left, y_next)
        basic_plot(node['right'], x_right, y_next)

def is_leaf(node):
    return node['left'] == 0

# Convert the tree to a form suitable for the Reingold-Tilford algorithm.
def pre_pass(node, max_depth):
    # Add new fields.
    node['x'] = 0
    node['mod'] = 0
    node['shift'] = 0
    # Consider left subtree.
    if type(node['left']) is int: # Convert left leaf node to dict.
        node['left'] = {'val' : node['left'], 'left' : 0, 'right' : 0, 'mod' : 0, 'shift' : 0, 'x' : 0, 'depth' : node['depth'] + 1}
    else:
        max_depth = max(max_depth, pre_pass(node['left'], max_depth))
    # Consider right subtree.
    if type(node['right']) is int: # Convert right leaf node to dict.
        node['right'] = {'val' : node['right'], 'left' : 0, 'right' : 0, 'mod' : 0, 'shift' : 0, 'x' : 0, 'depth' : node['depth'] + 1}
    else:
        max_depth = max(max_depth, pre_pass(node['right'], max_depth))
    return max(max_depth, node['depth'])

# Find a list of the leftmost elements of a subtree.
def find_left_contour(node, starting_depth, contour, total_mod):
    total_x = node['x'] + total_mod
    element = contour[node['depth'] - starting_depth]
    if element == -1 or element > total_x:
        contour[node['depth'] - starting_depth] = total_x
    total_mod += node['mod']
    if not is_leaf(node):
        find_left_contour(node['left'], starting_depth, contour, total_mod)
        find_left_contour(node['right'], starting_depth, contour, total_mod)

# Find a list of the rightmost elements of a subtree.
def find_right_contour(node, starting_depth, contour, total_mod):
    total_x = node['x'] + total_mod
    element = contour[node['depth'] - starting_depth]
    if element == -1 or element < total_x:
        contour[node['depth'] - starting_depth] = total_x
    total_mod += node['mod']
    if not is_leaf(node):
        find_right_contour(node['left'], starting_depth, contour, total_mod)
        find_right_contour(node['right'], starting_depth, contour, total_mod)

# First pass of the Reingold-Tilford algorithm. Compute preliminary values of x, mod and shift for each node.
def first_pass(node, max_depth):
    if is_leaf(node):
        return
    # Postorder traversal.
    first_pass(node['left'], max_depth)
    first_pass(node['right'], max_depth)
    # Consider left subtree.
    if is_leaf(node['left']):
        node['left']['x'] = 0
    else:
        node['left']['x'] = (node['left']['left']['x'] + node['left']['right']['x'] + node['left']['right']['shift']) / 2
    # Consider right subtree.
    node['right']['x'] = node['left']['x'] + 1
    if not is_leaf(node['right']):
        node['right']['mod'] = node['right']['x'] - (node['right']['left']['x'] + node['right']['right']['x'] + node['right']['right']['shift']) / 2
    # Find contours.
    starting_depth = node['depth'] + 1
    contour_size = max_depth - starting_depth + 1
    right_contour = [-1] * contour_size
    left_contour = [-1] * contour_size
    find_right_contour(node['left'], starting_depth, right_contour, 0)
    find_left_contour(node['right'], starting_depth, left_contour, 0)
    # Compute needed shift.
    max_shift = 0
    for i in range(contour_size):
        if left_contour[i] == -1 or right_contour[i] == -1: # Break if we have reached the end of one subtree.
            break
        shift = right_contour[i] - left_contour[i] + 1 # Leave at least 1 unit of space between nodes.
        if shift > max_shift:
            max_shift = shift
    node['right']['shift'] = max_shift

# Second pass of the Reingold-Tilford algorithm. Applies mod and shift to x values, and also records the min value of x (if negative).
def second_pass(node, total_mod, total_shift, min_x):
    # Apply shift but not mod to current x value.
    total_shift += node['shift']
    node['x'] += total_mod + total_shift
    if node['x'] < min_x: # Check if we have a min value of x.
        min_x = node['x']
    total_mod += node['mod']
    if not is_leaf(node):
        min_x_1 = second_pass(node['left'], total_mod, total_shift, min_x)
        min_x_2 = second_pass(node['right'], total_mod, total_shift, min_x)
        min_x = min(min_x_1, min_x_2)
    return min_x

# Third pass of the Reingold-Tilford algorithm. In the case of a negative x value, shifts the tree to make all x values positive.
def third_pass(node, adj):
    node['x'] += adj
    if not is_leaf(node):
        third_pass(node['left'], adj)
        third_pass(node['right'], adj)

# Plots the tree using computed x values and depths.
def recursive_plot(node, max_depth):
    if is_leaf(node):
        plt.text(node['x'], max_depth - node['depth'], str(node['val']))
    else:
        plt.text(node['x'], max_depth - node['depth'], str(node['split feature']) + ' < ' + str(node['split value']))
        plt.plot((node['left']['x'], node['x'], node['right']['x']), (max_depth - node['left']['depth'], max_depth - node['depth'], max_depth - node['right']['depth']))
        recursive_plot(node['left'], max_depth)
        recursive_plot(node['right'], max_depth)

def vis_tree(root1):
    plt.figure()
    plt.axis('off')
    root = copy.deepcopy(root1) # Make a copy of the tree since we will modify its structure in preprocessing.
    max_depth = pre_pass(root, 0) + 1
    first_pass(root, max_depth)
    root['x'] = (root['left']['x'] + root['right']['x'] + root['right']['shift']) / 2 # Centre the root node.
    min_x = second_pass(root, 0, 0, 0)
    third_pass(root, -min_x)
    recursive_plot(root, max_depth)
    plt.show()

if __name__ == '__main__':
    root = {'split feature' : 0, 'split value' : 5,
            'left' : {
                'split feature' : 1, 'split value' : 10, 'left' : 0, 'right' : {
                    'split feature' : 2, 'split value' : 50, 'left' : 5, 'right' : {
                        'split feature' : 3, 'split value' : 100, 'left' : {
                            'split feature' : 4, 'split value' : 300, 'left' : 0, 'right' : 1, 'depth' : 4
                        }, 'right' : 8, 'depth' : 3
                    }, 'depth' : 2
                }, 'depth' : 1
            },
            'right' : {
                'split feature' : 1, 'split value' : 20, 'left' : {
                    'split feature' : 2, 'split value' : 200, 'left' : 9, 'right' : 10, 'depth' : 2
                }, 'right' : 3, 'depth' : 1
            },
            'depth' : 0}
    
    # See basic plot first.
    plt.figure()
    plt.axis('off')
    basic_plot(root)

    # Comare to Reingold-Tilford algorithm.
    vis_tree(root)