import argparse
import torch
from color import BLUE, GREEN
from config import Config
from data_provider import DataProvider, LoadingMethod
from train import Train
from time import sleep
from pathlib import Path
from tokenizer import Tokenizer



class Chat(Train):
    def __init__(
        self,
        vocab_size: int,
        pad_id: int,
        tokenizer: Tokenizer,
        learning_rate: float = 0.002,
    ):
        super().__init__(vocab_size, pad_id, tokenizer, learning_rate)

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

        while True:
            try:
                q = input(GREEN + "You (输入 exit 退出): ")
            except (EOFError, KeyboardInterrupt):
                print()
                break
            if q.strip().lower() in {"exit", "quit", "退出"}:
                break
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

def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a trained AI-MODEL checkpoint.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-name", default="silver/lccc")
    parser.add_argument("--max-dialogs", type=int, default=10_000)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    args = parser.parse_args()

    config = Config(output_dir=args.output_dir.expanduser().resolve())
    checkpoint_path = (
        args.checkpoint.expanduser().resolve()
        if args.checkpoint is not None
        else config.checkpoint_dir / "last.ckpt"
    )
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not config.vocabulary_path.is_file():
        raise FileNotFoundError(f"Vocabulary does not exist: {config.vocabulary_path}")
    if not config.word2vec_path.is_file():
        raise FileNotFoundError(f"Word2Vec model does not exist: {config.word2vec_path}")

    data_provider = DataProvider(
        args.dataset_name,
        LoadingMethod.PATH,
        max_dialogs=args.max_dialogs,
    )
    tokenizer = Tokenizer(
        data_provider,
        is_new_dataset=False,
        max_sequence_length=args.max_sequence_length,
        vocabulary_path=config.vocabulary_path,
        word2vec_path=config.word2vec_path,
        allow_vocabulary_updates=False,
    )
    model = Chat(
        vocab_size=len(tokenizer.TRAIN_VOCABULARY),
        pad_id=tokenizer.SPECIAL_TOKEN["<PAD>"],
        tokenizer=tokenizer,
    )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.begin_chat()


if __name__ == "__main__":
    main()
    

    
