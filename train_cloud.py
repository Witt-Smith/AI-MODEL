import argparse
import os
from pathlib import Path
from typing import Optional, Union

import lightning as L
import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from torch.utils.data import DataLoader

from config import Config, PROJECT_ROOT, TrainingConfig
from data_provider import DataProvider, LoadingMethod
from tokenizer import Tokenizer
from train import DataSet, Train


def parse_devices(value: str) -> Union[str, int]:
    return int(value) if value.isdigit() else value


def build_parser() -> argparse.ArgumentParser:
    default_output_dir = Path(
        os.environ.get("AI_MODEL_OUTPUT_DIR", PROJECT_ROOT / "runs")
    )
    parser = argparse.ArgumentParser(
        description="Run resumable AI-MODEL training for a local or cloud job."
    )
    parser.add_argument("--output-dir", type=Path, default=default_output_dir)
    parser.add_argument("--dataset-name", default="silver/lccc")
    parser.add_argument("--max-dialogs", type=int, default=10_000)
    parser.add_argument("--max-sequence-length", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-epochs", type=int, default=1_000)
    parser.add_argument(
        "--continuous",
        action="store_true",
        help="Train without an epoch limit. Stop the cloud job to end training.",
    )
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint-every-n-epochs", type=int, default=1)
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--accelerator", default="auto")
    parser.add_argument("--devices", type=parse_devices, default="auto")
    parser.add_argument("--precision", default="32-true")
    parser.add_argument("--max-time", default=None)
    parser.add_argument("--resume-from", type=Path, default=None)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--fast-dev-run", action="store_true")
    return parser


def newest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    candidates = list(checkpoint_dir.glob("*.ckpt"))
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def resolve_checkpoint(args: argparse.Namespace, config: Config) -> Optional[Path]:
    if args.resume_from is not None:
        checkpoint = args.resume_from.expanduser().resolve()
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint}")
        return checkpoint
    if args.no_resume:
        return None
    return newest_checkpoint(config.checkpoint_dir)


def validate_checkpoint_contract(
    checkpoint_path: Optional[Path],
    vocabulary_size: int,
) -> None:
    if checkpoint_path is None:
        return
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    state_dict = checkpoint.get("state_dict", {})
    embedding_weight = state_dict.get("embedding.weight")
    linear_weight = state_dict.get("linear.weight")
    if embedding_weight is None or linear_weight is None:
        raise ValueError("Checkpoint is missing embedding or output-layer weights")
    if embedding_weight.size(0) != vocabulary_size:
        raise ValueError(
            "Checkpoint vocabulary size does not match vocabulary.json: "
            f"checkpoint={embedding_weight.size(0)}, vocabulary={vocabulary_size}"
        )
    if linear_weight.size(0) != vocabulary_size:
        raise ValueError(
            "Checkpoint output size does not match vocabulary.json: "
            f"checkpoint={linear_weight.size(0)}, vocabulary={vocabulary_size}"
        )


def main() -> None:
    args = build_parser().parse_args()
    max_epochs = -1 if args.continuous else args.max_epochs
    training_config = TrainingConfig(
        dataset_name=args.dataset_name,
        max_dialogs=args.max_dialogs,
        max_sequence_length=args.max_sequence_length,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        max_epochs=max_epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
        checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        gradient_clip_val=args.gradient_clip_val,
    )
    config = Config(
        output_dir=args.output_dir.expanduser().resolve(),
        training=training_config,
    )
    config.prepare()
    checkpoint_path = resolve_checkpoint(args, config)

    if checkpoint_path is not None:
        for artifact_path in (config.vocabulary_path, config.word2vec_path):
            if not artifact_path.is_file():
                raise FileNotFoundError(
                    "A checkpoint exists but a required training artifact is missing: "
                    f"{artifact_path}"
                )

    L.seed_everything(training_config.seed, workers=True)
    data_provider = DataProvider(
        training_config.dataset_name,
        LoadingMethod.PATH,
        max_dialogs=training_config.max_dialogs,
    )
    tokenizer = Tokenizer(
        data_provider,
        max_sequence_length=training_config.max_sequence_length,
        vocabulary_path=config.vocabulary_path,
        word2vec_path=config.word2vec_path,
        allow_vocabulary_updates=checkpoint_path is None,
    )
    vocabulary_size = len(tokenizer.TRAIN_VOCABULARY)
    validate_checkpoint_contract(checkpoint_path, vocabulary_size)

    dataset = DataSet(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=training_config.batch_size,
        shuffle=True,
        num_workers=training_config.num_workers,
        collate_fn=dataset.collate_fn,
        persistent_workers=training_config.num_workers > 0,
    )
    model = Train(
        vocab_size=vocabulary_size,
        pad_id=tokenizer.SPECIAL_TOKEN["<PAD>"],
        tokenizer=tokenizer,
        learning_rate=training_config.learning_rate,
    )
    if model.embedding.num_embeddings != vocabulary_size:
        raise ValueError("Embedding rows do not match vocabulary size")
    if model.linear.out_features != vocabulary_size:
        raise ValueError("Output layer does not match vocabulary size")

    logger = CSVLogger(str(config.log_dir), name="train")
    checkpoints = ModelCheckpoint(
        dirpath=config.checkpoint_dir,
        filename="epoch-{epoch:06d}",
        save_last=True,
        save_top_k=0,
        every_n_epochs=training_config.checkpoint_every_n_epochs,
        save_on_exception=True,
        auto_insert_metric_name=False,
    )
    trainer = L.Trainer(
        max_epochs=training_config.max_epochs,
        max_time=args.max_time,
        accelerator=args.accelerator,
        devices=args.devices,
        precision=args.precision,
        logger=logger,
        callbacks=[checkpoints],
        log_every_n_steps=training_config.log_every_n_steps,
        gradient_clip_val=training_config.gradient_clip_val,
        deterministic="warn",
        fast_dev_run=args.fast_dev_run,
        default_root_dir=config.output_dir,
    )

    print(f"output_dir={config.output_dir}")
    print(f"vocabulary_size={vocabulary_size}")
    print(f"word2vec={config.word2vec_path}")
    print(f"training_pairs={len(dataset)}")
    print(f"resume_from={checkpoint_path or 'none'}")
    print(f"max_epochs={training_config.max_epochs}")
    trainer.fit(
        model,
        train_dataloaders=dataloader,
        ckpt_path=str(checkpoint_path) if checkpoint_path is not None else None,
    )


if __name__ == "__main__":
    main()
