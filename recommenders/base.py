# recommenders/base.py
# Defines the abstract base class for all recommender systems.
# Lightly polished for documentation consistency and clarity.

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List
import pandas as pd


class BaseRecommender(ABC):
    """
    Abstract base class for recommender implementations.

    This allows consistent interaction between different recommendation
    algorithms (e.g., content-based, collaborative, hybrid) without changing
    the front-end or evaluation code.
    """

    @abstractmethod
    def get_items(self) -> List[str]:
        """
        Return a list of available item names (e.g., movie titles).

        Returns
        -------
        List[str]
            A list of unique item identifiers to display in the interface.
        """
        raise NotImplementedError

    @abstractmethod
    def recommend(self, item_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Given an item title, return a DataFrame with the top_n recommended items.

        Parameters
        ----------
        item_title : str
            The title (or unique identifier) of the reference item.
        top_n : int
            Number of similar items to return.

        Returns
        -------
        pd.DataFrame
            DataFrame containing recommended items and relevant metadata.
        """
        raise NotImplementedError
