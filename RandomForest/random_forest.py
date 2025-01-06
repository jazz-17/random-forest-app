
import numpy as np
import pandas as pd
from .decision_tree import DecisionTree
from collections import Counter
from typing import List

# Random Forest Classifier with Multiple Trees
class RandomForest:
    def __init__(self, n_trees: int, max_depth: int, min_size: int):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_size = min_size
        self.trees: List[DecisionTree] = []
        
    def fit(self, data: pd.DataFrame) -> None:
        """Train the random forest by building multiple decision trees."""
        for _ in range(self.n_trees):
            sample = self.bootstrap_sample(data)
            tree = DecisionTree(self.max_depth, self.min_size)
            tree.build_tree(sample.values)
            self.trees.append(tree)

    def bootstrap_sample(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create a bootstrap sample of the dataset."""
        n_samples: int = data.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return data.iloc[indices]
    
    # def predict(self, data: pd.DataFrame) -> List[int | float | str]:
    #     """Predict the class for all rows in the dataset."""
    #     return [self.predict_row(row) for row in data.values]
    # def predict(self, data: pd.DataFrame) -> List[int | float | str]:
    #     """Predict the class for all rows in the dataset."""
    #     return [self.predict_row(row).tolist() for row in data.values]
    
    def predict(self, data: pd.DataFrame) -> List[int | float | str]:
        """Predict the class for all rows in the dataset and convert predictions to native Python types."""
        return [self.predict_row(row).tolist() if isinstance(self.predict_row(row), np.generic) else self.predict_row(row) 
                for row in data.values]    
    
    def predict_row(self, row: np.ndarray) -> int | float | str:
        """Make a prediction for a single row using majority voting from all trees."""
        predictions = [tree.predict(tree.root, row) for tree in self.trees]
        return Counter(predictions).most_common(1)[0][0]


