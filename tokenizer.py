import json
from pathlib import Path
from typing import Optional

from jieba import cut

from data_provider import DataProvider
from wordvec import WordVec


class Tokenizer(WordVec):
    """Build a vocabulary shared by LCCC question and answer turns."""

    def __init__(
        self,
        data_provider: DataProvider,
        max_sequence_length: int = 64,
        vocabulary_path: Optional[Path] = None,
        word2vec_path: Optional[Path] = None,
        allow_vocabulary_updates: bool = True,
    ):
        if max_sequence_length < 3:
            raise ValueError("max_sequence_length must be at least 3")

        super().__init__(data_provider, model_path=word2vec_path)
        self.SPECIAL_TOKEN = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }
        self.MAX_SEQUENCE_LENGTH = max_sequence_length
        self.VOCABULARY_PATH = (
            Path(vocabulary_path)
            if vocabulary_path is not None
            else Path(__file__).resolve().parent / "data" / "vocabulary.json"
        )
        self.ALLOW_VOCABULARY_UPDATES = allow_vocabulary_updates
        self.build_vocabulary()
        self.TRAIN_VOCABULARY: dict[str, int] = self.read_vocabulary()

    def word_segmentation(self, text: str) -> list[str]:
        return list(cut(text.strip()))

    def build_vocabulary(self) -> None:
        vocabulary = dict(self.SPECIAL_TOKEN)

        if self.VOCABULARY_PATH.exists():
            self.VOCABULARY_PATH.unlink()

        if self.VOCABULARY_PATH.exists() and self.VOCABULARY_PATH.stat().st_size > 0:
            with self.VOCABULARY_PATH.open(mode="r", encoding="utf-8") as file:
                vocabulary = json.load(file)

        if not self.ALLOW_VOCABULARY_UPDATES:
            self.SPECIAL_TOKEN = vocabulary
            self.validate_vocabulary(vocabulary)
            return

        tokens = set(self.WORD2VEC.wv.key_to_index)

        for token in sorted(tokens):
            if token not in vocabulary:
                vocabulary[token] = len(vocabulary)

        self.SPECIAL_TOKEN = vocabulary
        self.validate_vocabulary(vocabulary)
        self.write_vocabulary()

    def validate_vocabulary(self, vocabulary: dict[str, int]) -> None:
        for token, expected_id in {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
        }.items():
            if vocabulary.get(token) != expected_id:
                raise ValueError(
                    f"Invalid special-token ID for {token}: "
                    f"expected {expected_id}, got {vocabulary.get(token)}"
                )

        token_ids = sorted(vocabulary.values())
        if token_ids != list(range(len(vocabulary))):
            raise ValueError("Vocabulary IDs must be unique and contiguous")

    def read_vocabulary(self) -> dict[str, int]:
        with self.VOCABULARY_PATH.open(mode="r", encoding="utf-8") as file:
            return json.load(file)

    def write_vocabulary(self) -> None:
        self.VOCABULARY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with self.VOCABULARY_PATH.open(mode="w", encoding="utf-8") as file:
            json.dump(self.SPECIAL_TOKEN, file, ensure_ascii=False)

    def get_questions(self) -> list[str]:
        return [question for question, _ in self.DATA_PROVIDER.get_pairs()]

    def get_answers(self) -> list[str]:
        return [answer for _, answer in self.DATA_PROVIDER.get_pairs()]

    def get_ids(self, words: str) -> list[int]:
        unk_id = self.TRAIN_VOCABULARY["<UNK>"]
        token_ids = [
            self.TRAIN_VOCABULARY.get(token, unk_id)
            for token in self.word_segmentation(words)[
                : self.MAX_SEQUENCE_LENGTH - 2
            ]
        ]
        return [
            self.TRAIN_VOCABULARY["<BOS>"],
            *token_ids,
            self.TRAIN_VOCABULARY["<EOS>"],
        ]

    def decode(self, token_ids) -> str:
        token_ids = token_ids.detach().cpu().tolist()
        ignored_ids = {
            self.SPECIAL_TOKEN["<PAD>"],
            self.SPECIAL_TOKEN["<BOS>"],
            self.SPECIAL_TOKEN["<EOS>"],
        }
        tokenid_to_token = {
            token_id: token
            for token, token_id in self.TRAIN_VOCABULARY.items()
        }
        return "".join(
            tokenid_to_token.get(token_id, "<UNK>")
            for token_id in token_ids
            if token_id not in ignored_ids
        )
