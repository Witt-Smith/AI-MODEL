
import pandas as pd
import torch
from color import GREEN
from data_provider import DataProvider, LoadingMethod
from train import DataSet, Train
from time import sleep
import lightning as L
from pathlib import Path
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from log import Log
from color import GREEN, YELLOW, BLUE, PURPLE, CYAN, RESET
from tokenizer import Tokenizer
from time import sleep



class Chat(Train):
    def __init__(self, vocab_size: int, pad_id: int, tokenizer: Tokenizer):
        super().__init__(vocab_size, pad_id, tokenizer)

    def generate(self,question: str,eos_id: int,bos_id: int) -> torch.Tensor:
        ''' Generate a response to the given question '''
        
        self.question = torch.tensor(
                        self.tokenizer.get_ids(question),
                        dtype = torch.long,
                        device = self.device,
                    ).unsqueeze(0)
        
        self.generated_ids = torch.full(
            (1, 1),
            fill_value = bos_id,
            dtype = torch.long,
            device = self.question.device
        )

        self.question_lengths = torch.tensor(
            [self.question.size(1)],
            dtype=torch.long,
        )
    
        with torch.inference_mode():
            for step in range(50):
                answer_logits = self(
                    self.question,
                    self.generated_ids,
                    self.question_lengths
                )

                self.next_id = answer_logits[:, -1, :].argmax(
                    dim=-1,
                    keepdim=True,
                )
                self.generated_ids = torch.cat(
                    [self.generated_ids, self.next_id],
                    dim = 1
                )

                if self.next_id.item() == eos_id:
                    break

        return self.generated_ids


    def begin_chat(self)-> None:
        self.eval()

        for _ in range(10):
            q = input(GREEN + "You: ")
            generated_ids = self.generate(
                q,
                self.tokenizer.SPECIAL_TOKEN["<EOS>"],
                self.tokenizer.SPECIAL_TOKEN["<BOS>"],
            )
            answer = self.tokenizer.decode(generated_ids[0])
            self.answer(answer)
            
    def answer(self, answer: str)-> any:
        print(BLUE + "AI : ",end = "")
        for i in answer: 
            print(i,flush = True,end = "")
            sleep(0.05)
        print()

if __name__ == "__main__":
    logger = CSVLogger(r'/Users/wittsmith/Desktop/AI-MODEL/logs',"train_log")
    log = Log(path = Path(logger.log_dir) / "metrics.csv")
    checkpoint_dir = Path(r"/Users/wittsmith/Desktop/AI-MODEL/checkpoints/last.ckpt")
    last_checkpoint = checkpoint_dir / "last.ckpt"
    callback = ModelCheckpoint(dirpath = checkpoint_dir,save_last = True)
    trainer = L.Trainer(max_epochs = 1000,logger = logger,log_every_n_steps = 1,callbacks = callback)
    data_provider = DataProvider(r"silver/lccc", LoadingMethod.PATH, max_dialogs = 10_000)
    tokenizer = Tokenizer(data_provider,False,max_sequence_length = 64)
    dataset = DataSet(tokenizer = tokenizer)
    train = Chat(vocab_size = len(tokenizer.TRAIN_VOCABULARY), pad_id = tokenizer.SPECIAL_TOKEN["<PAD>"], tokenizer = tokenizer)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size = 4, shuffle = True,num_workers = 0,collate_fn = dataset.collate_fn)
    trainer.fit(train,dataloader,ckpt_path = last_checkpoint if last_checkpoint.exists() else None)
    train.begin_chat()
    

    
