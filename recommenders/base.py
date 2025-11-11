# recommenders/base.py

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract base class for recommender implementations.

    This lets us swap algorithms later (e.g., different content-based variants)
    without changing the UI.
    """

    @abstractmethod
    def get_items(self) -> List[str]:
        """Return a list of available item names (e.g., movie titles)."""
        pass

    @abstractmethod
    def recommend(self, item_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Given an item title, return a DataFrame with the top_n recommended items.
        """
        pass
