from sympy import im
import torch
import torch.nn as nn
import lightning as L
import pandas as pd
from pathlib import Path
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from log import Log
from color import GREEN, YELLOW, BLUE, PURPLE, CYAN, RESET
from tokenizer import Tokenizer
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils.rnn import pack_padded_sequence
from wordvec import WordVec

class Train(L.LightningModule):
    def __init__(self,vocab_size: int, pad_id: int, tokenizer: Tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.loss_fn = nn.CrossEntropyLoss(ignore_index = pad_id)
        self.linear = nn.Linear(in_features = 128, out_features = vocab_size, bias = True)
        self.embedding = nn.Embedding.from_pretrained(
            self.tokenizer.get_vector(self.tokenizer.TRAIN_VOCABULARY),
            freeze=True
        )

        self.encoder = nn.GRU(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )

        self.decoder = nn.GRU(
            input_size=128,
            hidden_size=128,
            batch_first=True,
        )
        

    def forward(self, question_ids, decoder_input_ids, question_lengths):
        question_embedding = self.embedding(question_ids)
        decoder_embedding = self.embedding(decoder_input_ids)

        packed_question = pack_padded_sequence(
            question_embedding,
            question_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, encoder_hidden = self.encoder(packed_question)
        decoder_output, _ = self.decoder(
            decoder_embedding,
            encoder_hidden,
        )
        logits = self.linear(decoder_output)
        return logits

    def training_step(self, batch, batch_idx):
        x,y,question_lengths = batch
        # 标签分离
        decoder_input_ids = y[:, :-1]
        labels = y[:, 1:]

        prediction = self(x, decoder_input_ids,question_lengths)
        assert prediction.shape[:2] == labels.shape
        assert prediction.size(-1) == self.linear.out_features
        assert self.embedding.weight.shape == ((len(self.tokenizer.TRAIN_VOCABULARY)), 128), (
            "Embedding矩阵形状不一致："
            f"实际形状={tuple(self.embedding.weight.shape)}，"
            f"预期形状=({len(self.tokenizer.TRAIN_VOCABULARY)}, 128)，"
        )
        
        normal_loss = self.loss_fn(
            prediction.reshape(-1, prediction.size(-1)),
            labels.reshape(-1),
        )
        lr = self.optimizers().param_groups[0]['lr']

        self.log_dict(
            {
                "normal_loss": normal_loss,
                "lr" : lr
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            batch_size=x.size(0),
        )

        return normal_loss

    def configure_optimizers(self):
        return torch.optim.Adam(
            params = self.parameters(),
            lr = 0.002,
        )
    



class DataSet():
    def __init__(self,tokenizer: Tokenizer):
        self.tokenizer = tokenizer
        # 包含question and answer的问题.
        # 用于提供给__getitem__进行q,v进行查询.
        self.questions: list[str] = self.tokenizer.get_questions()
        self.answers: list[str] = self.tokenizer.get_answers()

    def __getitem__(self, key):
        # 当前batch的question_ids and answer_ids.
        # 调用tokenizer的方法获取token ID列表.
        question_ids: list[int] = self.tokenizer.get_ids(self.questions[key])
        answer_ids: list[int] = self.tokenizer.get_ids(self.answers[key])

        feature =torch.tensor(
                        question_ids,
                        dtype=torch.long,
                    )
        
        
        target = torch.tensor(
                        answer_ids,
                        dtype=torch.long,
                    )
        
        return feature,target
    
    def collate_fn(self, batch):
        features, targets = zip(*batch)

        organized_feature = pad_sequence(
            features,
            True,
            self.tokenizer.SPECIAL_TOKEN["<PAD>"]
            )
        organized_target = pad_sequence(
            targets,
            True,
            self.tokenizer.SPECIAL_TOKEN["<PAD>"]
            )
        question_lengths = torch.tensor(
            [len(feature) for feature in features],
            dtype=torch.int
        )

        return organized_feature,organized_target,question_lengths

    
    def __len__(self):
        return len(self.questions)
    
