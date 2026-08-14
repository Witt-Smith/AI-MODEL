from gensim.models import Word2Vec
import torch
from data_provider import DataProvider, LoadingMethod
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from numpy import ones
import numpy as np
from pathlib import Path
from typing import Optional

class WordVec():    
    ''' A class for handling word vector operations '''

    
    def __init__(
        self,
        data_provider: DataProvider,
        model_path: Optional[Path] = None,
    ):
        self.DATA_PROVIDER = data_provider
        self.WORD2VEC_PATH = Path(model_path) if model_path is not None else None
        if self.WORD2VEC_PATH is not None and self.WORD2VEC_PATH.is_file():
            self.WORD2VEC = Word2Vec.load(str(self.WORD2VEC_PATH))
        else:
            self.WORD2VEC = Word2Vec(
                self.DATA_PROVIDER.load_word_data(),
                vector_size = 128,
                min_count = 2,
                workers = 1 ,
                alpha = 0.002,
                epochs = 20
            )
            if self.WORD2VEC_PATH is not None:
                self.WORD2VEC_PATH.parent.mkdir(parents=True, exist_ok=True)
                self.WORD2VEC.save(str(self.WORD2VEC_PATH))
        if self.WORD2VEC.vector_size != 128:
            raise ValueError(
                f"Word2Vec vector size must be 128, got {self.WORD2VEC.vector_size}"
            )
        self.PCA = PCA(n_components = 2)

    def train(self, train_file: str)-> None:
        self.WORD2VEC.train(corpus_file = train_file)

    def get_vector(self,vocabulary: dict[str, int]) -> torch.Tensor:
        vector_size = self.WORD2VEC.vector_size
        embedding_matrix = torch.empty(
            len(vocabulary),
            vector_size,
            dtype=torch.float,
        )
        torch.nn.init.normal_(
            embedding_matrix,
            mean=0.0,
            std=0.02,
        )
        pad_id = vocabulary["<PAD>"]
        embedding_matrix[pad_id].zero_()
        unk_id = vocabulary["<UNK>"]
        word_vectors = torch.tensor(
            self.WORD2VEC.wv.vectors,
            dtype=torch.float,
        )
        embedding_matrix[unk_id] = word_vectors.mean(
            dim=0
        )
        for token, token_id in vocabulary.items():
            if token in self.WORD2VEC.wv.key_to_index:
                embedding_matrix[token_id] = torch.tensor(
                    self.WORD2VEC.wv[token],
                    dtype=torch.float,
                )

        return embedding_matrix
    
    def semantic_map(self,*words: str)-> None:
        ''' Show semantic map '''

        self.all_points = self.PCA.fit_transform(
            self.WORD2VEC.wv.get_normed_vectors()
        )
        self.words_points: list[int] = [self.WORD2VEC.wv.key_to_index[word] for word in words]
        plt.figure(
            figsize = (12.0,9.0),
        )
        plt.scatter(
            x = self.all_points[:, 0],
            y = self.all_points[:, 1],
            s = 23,
            alpha = 0.01,
            color = "black"
        )
        plt.scatter (
            x = self.all_points[self.words_points[0], 0],
            y = self.all_points[self.words_points[0], 1],
            s = 25,
            alpha = 1,
            color = "red"
        )
        plt.show()


    
    
    def most_similar(self, word: str, topn: int = 5)-> list[tuple[str, float]]:
        return self.WORD2VEC.wv.most_similar(word, topn=topn)

    def relationship_map(self, *words: str) -> None:
        matrix = np.array([
            [
                self.WORD2VEC.wv.similarity(word1, word2)
                for word2 in words
            ]
            for word1 in words
        ])

        plt.rcParams["font.family"] = "Arial Unicode MS"
        plt.rcParams["axes.unicode_minus"] = False

        figure, axis = plt.subplots(figsize=(9, 7))
        image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)

        axis.set_xticks(range(len(words)), labels=words, rotation=45)
        axis.set_yticks(range(len(words)), labels=words)

        axis.set_xlabel("对比词 word2")
        axis.set_ylabel("基准词 word1")
        axis.set_title("词语关系程度")

        for row, column in np.ndindex(matrix.shape):
            axis.text(
                column,
                row,
                f"{words[row]} ↔ {words[column]}\n"
                f"{matrix[row, column]:.2f}",
                ha="center",
                va="center",
            )

        figure.colorbar(image, ax=axis, label="余弦相似度")
        figure.tight_layout()
        plt.show()
