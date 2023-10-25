import copy
import random
import matplotlib.pyplot as plt

SIBLING_SEP = 100 # Minimum separation between siblings.
SUBTREE_SEP = 100 # Minimum separation between the closest nodes of two adjacent subtrees.
LAYER_SEP = 100 # Separation between adjacent layers.
MIN_COLOUR = 0.2 # Minimum value for RGB when plotting lines.
MAX_COLOUR = 1.0 # Maximum value for RGB when plotting lines.

# Naive solution using preorder recursion. Looks bad with deep trees.
# node: root node of the tree.
def basic_plot(node, x=0, y=50):
    if type(node) is int: # Plot leaf node
        plt.text(x, y, str(node), ha='center', va='center',
                 bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
    else:
        plt.text(x, y, str(node['split feature']) + ' < ' + str(node['split value']), ha='center', va='center',
                 bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
        y_next = y - 10
        x_left = x - 2**(1 - node['depth'])
        x_right = x + 2**(1 - node['depth'])
        plt.plot((x - 2**(1 - node['depth']), x, x + 2**(1 - node['depth'])), (y_next, y, y_next)) # Plot lines to children
        basic_plot(node['left'], x_left, y_next)
        basic_plot(node['right'], x_right, y_next)

# Returns true if the node is a leaf, otherwise false.
def is_leaf(node):
    return node['left'] == 0

# Convert the tree to a form suitable for the Reingold-Tilford algorithm.
# node: root node of the tree.
# Returns the maximum depth of the tree.
def pre_pass(node, max_depth=0):
    # Add new fields.
    node['x'] = 0
    node['mod'] = 0
    node['shift'] = 0
    # Consider left subtree.
    if type(node['left']) is int: # Convert left leaf node to dict
        node['left'] = {'val' : node['left'], 'left' : 0, 'right' : 0, 'mod' : 0, 'shift' : 0, 'x' : 0, 'depth' : node['depth'] + 1}
    else:
        max_depth = max(max_depth, pre_pass(node['left'], max_depth))
    # Consider right subtree.
    if type(node['right']) is int: # Convert right leaf node to dict
        node['right'] = {'val' : node['right'], 'left' : 0, 'right' : 0, 'mod' : 0, 'shift' : 0, 'x' : 0, 'depth' : node['depth'] + 1}
    else:
        max_depth = max(max_depth, pre_pass(node['right'], max_depth))
    return max(max_depth, node['depth'])

# Find a list of the leftmost or rightmost elements of a subtree.
# node: root node of subtree.
# starting_depth: the depth of the root node.
# contour: a list representing the contour so far.
# total_mod: a running count of the addition to the x value from mod.
# total_shift: a running count of the addition to the x value from shift.
# is_right: find right contour if true, otherwise find left contour.
def find_contour(node, starting_depth, contour, total_mod, total_shift, is_right):
    # Update shift, find new x, then update mod.
    total_shift += node['shift']
    total_x = node['x'] + total_mod + total_shift
    total_mod += node['mod']
    element = contour[node['depth'] - starting_depth]
    if element == -1 or (is_right and element < total_x) or (not is_right and element > total_x): # Found new node for contour
        contour[node['depth'] - starting_depth] = total_x
    if not is_leaf(node):
        find_contour(node['left'], starting_depth, contour, total_mod, total_shift, is_right)
        find_contour(node['right'], starting_depth, contour, total_mod, total_shift, is_right)

# First pass of the Reingold-Tilford algorithm. Compute preliminary values of x, mod and shift for each node.
# node: root node of the tree.
# max_depth: the maximum depth of the tree.
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
    node['right']['x'] = node['left']['x'] + SIBLING_SEP
    if not is_leaf(node['right']):
        node['right']['mod'] = node['right']['x'] - (node['right']['left']['x'] + node['right']['right']['x'] + node['right']['right']['shift']) / 2
    # Find contours.
    starting_depth = node['depth'] + 1
    contour_size = max_depth - starting_depth + 1
    right_contour = [-1] * contour_size
    left_contour = [-1] * contour_size
    find_contour(node['left'], starting_depth, right_contour, 0, 0, True)
    find_contour(node['right'], starting_depth, left_contour, 0, 0, False)
    # Compute needed shift.
    max_shift = 0
    for i in range(contour_size):
        if left_contour[i] == -1 or right_contour[i] == -1: # Break if we have reached the end of one subtree.
            break
        shift = right_contour[i] - left_contour[i] + SUBTREE_SEP # Leave space between nodes.
        if shift > max_shift:
            max_shift = shift
    node['right']['shift'] = max_shift

# Second pass of the Reingold-Tilford algorithm. Applies mod and shift to x values, and also records the min value of x (if negative).
# node: root node of the tree.
# total_mod: running count of the addition to the x value from mod.
# total_shift: running count of the addition to the x value from shift.
# min_x: the minimum x value found so far.
# Returns minimum x value.
def second_pass(node, total_mod, total_shift, min_x):
    # Apply shift, check min x, then apply mod.
    total_shift += node['shift']
    node['x'] += total_mod + total_shift
    if node['x'] < min_x:
        min_x = node['x']
    total_mod += node['mod']
    if not is_leaf(node):
        min_x_1 = second_pass(node['left'], total_mod, total_shift, min_x)
        min_x_2 = second_pass(node['right'], total_mod, total_shift, min_x)
        min_x = min(min_x_1, min_x_2)
    return min_x

# Third pass of the Reingold-Tilford algorithm. Shifts the tree to make all x values positive.
# node: root node of the tree.
# adj: how much to adjust each x value by.
def third_pass(node, adj):
    node['x'] += adj
    if not is_leaf(node):
        third_pass(node['left'], adj)
        third_pass(node['right'], adj)

# Plots the tree using computed x values and depths.
# node: root node of the tree.
# max_depth: the maximum depth of the tree.
def recursive_plot(node, max_depth, colours):
    if is_leaf(node):
        plt.text(node['x'], LAYER_SEP * (max_depth - node['depth']), str(node['val']),
                 ha='center', va='center',
                 bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
    else:
        plt.text(node['x'], LAYER_SEP * (max_depth - node['depth']), str(node['split feature']) + ' < ' + str(node['split value']),
                 ha='center', va='center',
                 bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
        plt.plot((node['left']['x'], node['x'], node['right']['x']),
                 (LAYER_SEP * (max_depth - node['left']['depth']), LAYER_SEP * (max_depth - node['depth']), LAYER_SEP * (max_depth - node['right']['depth'])),
                 color=colours[node['depth']])
        recursive_plot(node['left'], max_depth, colours)
        recursive_plot(node['right'], max_depth, colours)

# Visualise the tree.
# root: root node of the tree.
def vis_tree(root):
    # Set up plot.
    plt.figure()
    plt.axis('off')

    # Apply Reingold-Tilford algorithm.
    new_root = copy.deepcopy(root) # Make a copy of the tree since we will modify its structure in preprocessing
    max_depth = pre_pass(new_root) + 1 # Add one since max_depth refers to node before leaf node
    first_pass(new_root, max_depth)
    new_root['x'] = (new_root['left']['x'] + new_root['right']['x'] + new_root['right']['shift']) / 2 # Centre the root node
    min_x = second_pass(new_root, 0, 0, 0)
    if min_x < 0: # Only bother adjusting if we have a negative x value somewhere
        third_pass(new_root, -min_x)

    # Draw plot.
    colours = []
    for i in range(max_depth + 1):
        # val = min(i / max_depth, 1)
        # colours.append((val, val, val))
        # Generate some random colours for each level of edges.
        colours.append((random.uniform(MIN_COLOUR, MAX_COLOUR), random.uniform(MIN_COLOUR, MAX_COLOUR), random.uniform(MIN_COLOUR, MAX_COLOUR)))
    recursive_plot(new_root, max_depth, colours)
    plt.show()