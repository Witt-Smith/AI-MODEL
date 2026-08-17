import json
from pathlib import Path
from typing import Iterable, Optional, Union

import torch
from jieba import cut

from data_provider import DataProvider
from wordvec import WordVec

from config import PROJECT_OUTPUT_DIR


class Tokenizer(WordVec):
    """Segment text and keep token IDs aligned with training artifacts."""

    SPECIAL_TOKEN_IDS = {
        "<PAD>": 0,
        "<UNK>": 1,
        "<BOS>": 2,
        "<EOS>": 3,
    }

    def __init__(
        self,
        data_provider: DataProvider,
        max_sequence_length: int,
        vocabulary_path: Optional[Path] = None,
        word2vec_path: Optional[Path] = None,
        allow_vocabulary_updates: bool = True,
    ):
        if max_sequence_length < 3:
            raise ValueError("max_sequence_length must be at least 3")
        super().__init__(data_provider, model_path=word2vec_path)
        self.max_sequence_length = max_sequence_length
        self.vocabulary_path = (
            vocabulary_path
            if vocabulary_path is not None
            else PROJECT_OUTPUT_DIR / "artifacts" / "vocabulary.json"
        )
        self.allow_vocabulary_updates = allow_vocabulary_updates
        self.token_to_id = self._build_vocabulary()
        self.id_to_token = {
            token_id: token for token, token_id in self.token_to_id.items()
        }
        self.pad_id = self.SPECIAL_TOKEN_IDS["<PAD>"]
        self.unk_id = self.SPECIAL_TOKEN_IDS["<UNK>"]
        self.bos_id = self.SPECIAL_TOKEN_IDS["<BOS>"]
        self.eos_id = self.SPECIAL_TOKEN_IDS["<EOS>"]
        self._ignored_decode_ids = frozenset({self.pad_id, self.bos_id})

    def word_segmentation(self, text: str) -> list[str]:
        if not isinstance(text, str):
            raise TypeError("text must be a string")
        return list(cut(text.strip()))

    def _build_vocabulary(self) -> dict[str, int]:
        if self.vocabulary_path.is_file():
            if self.vocabulary_path.stat().st_size == 0:
                raise ValueError(
                    f"Vocabulary file is empty: {self.vocabulary_path}"
                )
            vocabulary = self._read_vocabulary()
        elif not self.allow_vocabulary_updates:
            raise FileNotFoundError(
                "Vocabulary updates are disabled but no vocabulary exists: "
                f"{self.vocabulary_path}"
            )
        else:
            vocabulary = dict(self.SPECIAL_TOKEN_IDS)

        if self.allow_vocabulary_updates:
            for token in sorted(self.word2vec.wv.key_to_index):
                if token not in vocabulary:
                    vocabulary[token] = len(vocabulary)

        self.validate_vocabulary(vocabulary)
        if self.allow_vocabulary_updates:
            self._write_vocabulary(vocabulary)
        return vocabulary

    def validate_vocabulary(self, vocabulary: dict[str, int]) -> None:
        if not isinstance(vocabulary, dict) or not vocabulary:
            raise ValueError("Vocabulary must be a non-empty JSON object")

        for token, token_id in vocabulary.items():
            if not isinstance(token, str):
                raise TypeError("Vocabulary tokens must be strings")
            if not isinstance(token_id, int):
                raise TypeError(f"Vocabulary ID for {token!r} must be an integer")

        for token, expected_id in self.SPECIAL_TOKEN_IDS.items():
            actual_id = vocabulary.get(token)
            if actual_id != expected_id:
                raise ValueError(
                    f"Invalid special-token ID for {token}: "
                    f"expected {expected_id}, got {actual_id}"
                )

        token_ids = sorted(vocabulary.values())
        if token_ids != list(range(len(vocabulary))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")

    def _read_vocabulary(self) -> dict[str, int]:
        with self.vocabulary_path.open(mode="r", encoding="utf-8") as file:
            return json.load(file)

    def _write_vocabulary(self, vocabulary: dict[str, int]) -> None:
        self.vocabulary_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.vocabulary_path.with_suffix(
            self.vocabulary_path.suffix + ".tmp"
        )
        with temporary_path.open(mode="w", encoding="utf-8") as file:
            json.dump(vocabulary, file, ensure_ascii=False)
        temporary_path.replace(self.vocabulary_path)

    def get_ids(self, text: str) -> list[int]:
        token_to_id = self.token_to_id
        token_ids = [
            token_to_id.get(token, self.unk_id)
            for token in self.word_segmentation(text)[
                : self.max_sequence_length - 2
            ]
        ]
        return [
            self.bos_id,
            *token_ids,
            self.eos_id,
        ]

    def decode(
        self,
        token_ids: Union[Iterable[int], torch.Tensor],
    ) -> str:
        if isinstance(token_ids, torch.Tensor):
            decoded_ids = token_ids.detach().cpu().flatten().tolist()
        else:
            decoded_ids = list(token_ids)

        tokens: list[str] = []

        for token_id in decoded_ids:
            if isinstance(token_id, bool) or not isinstance(token_id, int):
                raise TypeError("token_ids must contain integers")
            if token_id == self.eos_id:
                break
            if token_id not in self._ignored_decode_ids:
                tokens.append(self.id_to_token.get(token_id, "<UNK>"))

        return "".join(tokens)
