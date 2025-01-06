import numpy as np
from collections import Counter
from typing import  Tuple, Optional

# TreeNode Class for Decision Tree Structure
class TreeNode:
    """Class representing a node in a decision tree."""
    def __init__(
        self,
        index: Optional[int] = None,
        value: Optional[float] = None,
        groups: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        left: Optional['TreeNode'] = None,
        right: Optional['TreeNode'] = None,
        is_leaf: bool = False,
        label: Optional[int| float| str] = None
    ):
        self.index = index  # Feature index for split
        self.value = value  # Threshold value for split
        self.groups = groups  # Left and right split groups
        self.left = left  # Left child node
        self.right = right  # Right child node
        self.is_leaf = is_leaf  # Indicates if the node is a leaf
        self.label = label  # Class label if it's a leaf node

# Decision Tree Classifier
class DecisionTree:
    """Decision Tree classifier for individual trees within a Random Forest."""
    def __init__(self, max_depth: int, min_size: int):
        self.max_depth = max_depth
        self.min_size = min_size
        self.root: Optional[TreeNode] = None

    def gini_impurity(self, groups: Tuple[np.ndarray, np.ndarray], classes: np.ndarray) -> float:
        """Calculate Gini impurity for a given split."""
        total_samples: float = float(sum([len(group) for group in groups]))
        gini: float = 0.0
        for group in groups:
            size: int = len(group)
            if size == 0:
                continue
            score: float = 0.0
            group_labels: np.ndarray = group[:, -1]
            for class_val in classes:
                proportion: float = np.sum(group_labels == class_val) / size
                score += proportion ** 2
            gini += (1.0 - score) * (size / total_samples)
        return gini

    def test_split(self, index: int, value: float, dataset: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Split the dataset based on a feature value."""
        left: np.ndarray = dataset[dataset[:, index] < value]
        right: np.ndarray = dataset[dataset[:, index] >= value]
        return left, right

    def get_best_split(self, dataset: np.ndarray) -> TreeNode:
        """Find the best feature and value to split the dataset."""
        class_values: np.ndarray = np.unique(dataset[:, -1])
        best_index: Optional[int] = None
        best_value: Optional[float] = None
        best_score: float = float('inf')
        best_groups: Optional[Tuple[np.ndarray, np.ndarray]] = None
        
        for index in range(dataset.shape[1] - 1):
            for row in dataset:
                groups: Tuple[np.ndarray, np.ndarray] = self.test_split(index, row[index], dataset)
                gini: float = self.gini_impurity(groups, class_values)
                if gini < best_score:
                    best_index, best_value, best_score, best_groups = index, row[index], gini, groups
        return TreeNode(index=best_index, value=best_value, groups=best_groups)

    def to_terminal(self, group: np.ndarray) -> int| float| str:
        """Create a terminal node with the most common class label in a group."""
        outcomes: np.ndarray = group[:, -1]
        return Counter(outcomes).most_common(1)[0][0]

    def split(self, node: TreeNode, depth: int) -> None:
        """Recursive function to build the tree by splitting nodes."""
        left, right = node.groups
        del node.groups  # No need to keep groups once the node is split
        if not left.any() or not right.any():
            node.left = node.right = TreeNode(is_leaf=True, label=self.to_terminal(np.vstack((left, right))))
            return
        if depth >= self.max_depth:
            node.left = TreeNode(is_leaf=True, label=self.to_terminal(left))
            node.right = TreeNode(is_leaf=True, label=self.to_terminal(right))
            return
        if len(left) <= self.min_size:
            node.left = TreeNode(is_leaf=True, label=self.to_terminal(left))
        else:
            node.left = self.get_best_split(left)
            self.split(node.left, depth + 1)
        if len(right) <= self.min_size:
            node.right = TreeNode(is_leaf=True, label=self.to_terminal(right))
        else:
            node.right = self.get_best_split(right)
            self.split(node.right, depth + 1)

    def build_tree(self, train: np.ndarray) -> None:
        """Build the decision tree."""
        self.root = self.get_best_split(train)
        self.split(self.root, 1)

    def predict(self, node: TreeNode, row: np.ndarray) -> int| float| str:
        """Make a prediction with the decision tree."""
        if node.is_leaf:
            return node.label
        if row[node.index] < node.value:
            return self.predict(node.left, row)
        else:
            return self.predict(node.right, row)
