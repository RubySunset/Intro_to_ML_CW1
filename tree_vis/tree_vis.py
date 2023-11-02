import copy
import random
# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Node structure:
# split feature: int
# split value: float
# left: dict
# right: dict
# depth: int
#
# Note: leaf nodes are represented as a numpy float (need to round to int)

class TreeVis:

    SIBLING_SEP = 100 # Minimum separation between siblings.
    SUBTREE_SEP = 100 # Minimum separation between the closest nodes of two adjacent subtrees.
    LAYER_SEP = 100 # Separation between adjacent layers.
    MIN_COLOUR = 0.0 # Minimum value for RGB when plotting lines.
    MAX_COLOUR = 0.8 # Maximum value for RGB when plotting lines.

    def __init__(self):
        pass

    # Returns true if the node is a leaf, otherwise false.
    def is_leaf(self, node):
        return node['left'] == 0

    # Convert the tree to a form suitable for the Reingold-Tilford algorithm.
    # node: root node of the tree.
    # Returns the maximum depth of the tree.
    def pre_pass(self, node, max_depth=0):
        # Add new fields.
        node['x'] = 0
        node['mod'] = 0
        node['shift'] = 0
        # Consider left subtree.
        if not type(node['left']) is dict: # Convert left leaf node to dict
            node['left'] = {'val' : round(node['left']), 'left' : 0, 'right' : 0,
                            'mod' : 0, 'shift' : 0, 'x' : 0, 'depth' : node['depth'] + 1}
        else:
            max_depth = max(max_depth, self.pre_pass(node['left'], max_depth))
        # Consider right subtree.
        if not type(node['right']) is dict: # Convert right leaf node to dict
            node['right'] = {'val' : round(node['right']), 'left' : 0, 'right' : 0,
                             'mod' : 0, 'shift' : 0, 'x' : 0, 'depth' : node['depth'] + 1}
        else:
            max_depth = max(max_depth, self.pre_pass(node['right'], max_depth))
        return max(max_depth, node['depth'])

    # Find a list of the leftmost or rightmost elements of a subtree.
    # node: root node of subtree.
    # starting_depth: the depth of the root node.
    # contour: a list representing the contour so far.
    # total_mod: a running count of the addition to the x value from mod.
    # total_shift: a running count of the addition to the x value from shift.
    # is_right: find right contour if true, otherwise find left contour.
    def find_contour(self, node, starting_depth, contour, total_mod, total_shift, is_right):
        # Update shift, find new x, then update mod.
        total_shift += node['shift']
        total_x = node['x'] + total_mod + total_shift
        total_mod += node['mod']
        element = contour[node['depth'] - starting_depth]
        if element == -1 or (is_right and element < total_x) or (not is_right and element > total_x): # Found new node for contour
            contour[node['depth'] - starting_depth] = total_x
        if not self.is_leaf(node):
            self.find_contour(node['left'], starting_depth, contour, total_mod, total_shift, is_right)
            self.find_contour(node['right'], starting_depth, contour, total_mod, total_shift, is_right)

    # First pass of the Reingold-Tilford algorithm. Compute preliminary values of x, mod and shift for each node.
    # node: root node of the tree.
    # max_depth: the maximum depth of the tree.
    def first_pass(self, node, max_depth):
        if self.is_leaf(node):
            return
        left = node['left']
        right = node['right']
        # Postorder traversal.
        self.first_pass(left, max_depth)
        self.first_pass(right, max_depth)
        # Consider left subtree.
        if self.is_leaf(left):
            left['x'] = 0
        else:
            left['x'] = (left['left']['x'] + left['right']['x'] + left['right']['shift']) / 2
        # Consider right subtree.
        right['x'] = node['left']['x'] + self.SIBLING_SEP
        if not self.is_leaf(right):
            right['mod'] = right['x'] - (right['left']['x'] + right['right']['x'] + right['right']['shift']) / 2
        # Find contours.
        starting_depth = node['depth'] + 1
        contour_size = max_depth - starting_depth + 1
        right_contour = [-1] * contour_size
        left_contour = [-1] * contour_size
        self.find_contour(left, starting_depth, right_contour, 0, 0, True)
        self.find_contour(right, starting_depth, left_contour, 0, 0, False)
        # Compute needed shift.
        max_shift = 0
        for i in range(contour_size):
            if left_contour[i] == -1 or right_contour[i] == -1: # Break if we have reached the end of one subtree.
                break
            shift = right_contour[i] - left_contour[i] + self.SUBTREE_SEP # Leave space between nodes.
            if shift > max_shift:
                max_shift = shift
        right['shift'] = max_shift

    # Second pass of the Reingold-Tilford algorithm. Applies mod and shift to x values, and also records the min value of x (if negative).
    # node: root node of the tree.
    # total_mod: running count of the addition to the x value from mod.
    # total_shift: running count of the addition to the x value from shift.
    # min_x: the minimum x value found so far.
    # Returns minimum x value.
    def second_pass(self, node, total_mod, total_shift, min_x):
        # Apply shift, check min x, then apply mod.
        total_shift += node['shift']
        node['x'] += total_mod + total_shift
        if node['x'] < min_x:
            min_x = node['x']
        total_mod += node['mod']
        if not self.is_leaf(node):
            min_x_1 = self.second_pass(node['left'], total_mod, total_shift, min_x)
            min_x_2 = self.second_pass(node['right'], total_mod, total_shift, min_x)
            min_x = min(min_x_1, min_x_2)
        return min_x

    # Third pass of the Reingold-Tilford algorithm. Shifts the tree to make all x values positive.
    # node: root node of the tree.
    # adj: how much to adjust each x value by.
    def third_pass(self, node, adj):
        node['x'] += adj
        if not self.is_leaf(node):
            self.third_pass(node['left'], adj)
            self.third_pass(node['right'], adj)

    # Plots the tree using computed x values and depths.
    # node: root node of the tree.
    # max_depth: the maximum depth of the tree.
    def recursive_plot(self, node, max_depth, colours):
        if self.is_leaf(node):
            plt.text(node['x'], self.LAYER_SEP * (max_depth - node['depth']), 'leaf:' + str(node['val']),
                    ha='center', va='center', size='small',
                    bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
        else:
            plt.text(node['x'], self.LAYER_SEP * (max_depth - node['depth']),
                     '[X' + str(node['split feature']) + ' < ' + str(node['split value']) + ']',
                     ha='center', va='center', size='small',
                     bbox={'boxstyle' : 'square', 'ec' : (0, 0, 0), 'fc' : (1, 1, 1)})
            plt.plot((node['left']['x'], node['x'], node['right']['x']),
                     (self.LAYER_SEP * (max_depth - node['left']['depth']),
                     self.LAYER_SEP * (max_depth - node['depth']), self.LAYER_SEP * (max_depth - node['right']['depth'])),
                     color=colours[node['depth']])
            self.recursive_plot(node['left'], max_depth, colours)
            self.recursive_plot(node['right'], max_depth, colours)

    # Visualise the tree.
    # root: root node of the tree.
    def draw(self, root):
        # Set up plot.
        plt.figure()
        plt.axis('off')

        # Apply Reingold-Tilford algorithm.
        new_root = copy.deepcopy(root) # Make a copy of the tree since we will modify its structure in preprocessing
        max_depth = self.pre_pass(new_root) + 1 # Add one since max_depth refers to node before leaf node
        self.first_pass(new_root, max_depth)
        new_root['x'] = (new_root['left']['x'] + new_root['right']['x'] + new_root['right']['shift']) / 2 # Centre the root node
        min_x = self.second_pass(new_root, 0, 0, 0)
        if min_x < 0: # Only bother adjusting if we have a negative x value somewhere
            self.third_pass(new_root, -min_x)

        # Draw plot.
        colours = []
        for i in range(max_depth + 1):
            # val = min(i / max_depth, 1)
            # colours.append((val, val, val))
            # Generate some random colours for each level of edges.
            # colours.append((0.8, 0.8, 0.8))
            colours.append((random.uniform(self.MIN_COLOUR, self.MAX_COLOUR), random.uniform(self.MIN_COLOUR, self.MAX_COLOUR), random.uniform(self.MIN_COLOUR, self.MAX_COLOUR)))
        self.recursive_plot(new_root, max_depth, colours)
        plt.show()