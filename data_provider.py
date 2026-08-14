from enum import Enum
from functools import lru_cache
from typing import Optional

from datasets import DatasetDict, load_dataset
from jieba import cut


class LoadingMethod(Enum):
    PATH = "path"
    URL = "url"


class DataProvider:
    """Load LCCC dialogs and turn adjacent messages into QA pairs."""

    def __init__(
        self,
        dataset: str,
        loading_method: LoadingMethod,
        max_dialogs: int = 10_000,
    ):
        if max_dialogs <= 0:
            raise ValueError("max_dialogs must be greater than 0")

        self.dataset = dataset
        self._ds: Optional[DatasetDict] = None
        self.Loading_method = loading_method
        self.max_dialogs = max_dialogs

    @property
    def ds(self) -> DatasetDict:
        if self._ds is None:
            self._ds = load_dataset(self.dataset, "base")
        return self._ds

    @staticmethod
    def clean_turn(text: str) -> str:
        return "".join(str(text).split())

    @lru_cache(maxsize=1)
    def get_dialogs(self) -> list[list[str]]:
        if self.Loading_method != LoadingMethod.PATH:
            raise NotImplementedError("Only PATH loading is implemented")

        train_split = self.ds["train"]
        sample_count = min(self.max_dialogs, len(train_split))
        dialogs: list[list[str]] = []

        for row in train_split.select(range(sample_count)):
            turns: list[str] = []
            for turn in row["dialog"]:
                cleaned_turn = self.clean_turn(turn)
                if cleaned_turn:
                    turns.append(cleaned_turn)
            if turns:
                dialogs.append(turns)

        if not dialogs:
            raise ValueError("No dialogs were found in the dataset")

        return dialogs

    @lru_cache(maxsize=1)
    def get_pairs(self) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []

        for turns in self.get_dialogs():
            pairs.extend(zip(turns[:-1], turns[1:]))

        if not pairs:
            raise ValueError("No adjacent dialog pairs were found in the dataset")

        return pairs

    @lru_cache(maxsize=1)
    def load_word_data(self) -> list[list[str]]:
        return [
            list(cut(turn))
            for dialog in self.get_dialogs()
            for turn in dialog
        ]

    def get_ds(self) -> DatasetDict:
        return self.ds

    @lru_cache(maxsize=1)
    def dataset_load(self) -> DatasetDict:
        return self.ds
