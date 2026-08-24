from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


PROJECT_ROOT = Path(__file__).resolve().parent
PROJECT_OUTPUT_DIR = PROJECT_ROOT / "runs"


@dataclass(frozen=True)
class TrainingConfig:
    dataset_name: str = "silver/lccc"
    dataset_config: Optional[str] = "base"
    dialog_field: str = "dialog"
    max_dialogs: int = 10_000
    max_sequence_length: int = 64
    batch_size: int = 4
    num_workers: int = 0
    max_epochs: int = 1_000
    learning_rate: float = 0.002
    seed: int = 42
    checkpoint_every_n_epochs: int = 1
    log_every_n_steps: int = 1
    gradient_clip_val: float = 1.0

    def validate(self) -> None:
        if not self.dataset_name.strip():
            raise ValueError("dataset_name cannot be empty")
        if self.dataset_config is not None and not self.dataset_config.strip():
            raise ValueError("dataset_config cannot be empty")
        if not self.dialog_field.strip():
            raise ValueError("dialog_field cannot be empty")
        if self.max_dialogs <= 0:
            raise ValueError("max_dialogs must be greater than 0")
        if self.max_sequence_length < 3:
            raise ValueError("max_sequence_length must be at least 3")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.max_epochs == 0 or self.max_epochs < -1:
            raise ValueError("max_epochs must be -1 or greater than 0")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0")
        if self.checkpoint_every_n_epochs <= 0:
            raise ValueError("checkpoint_every_n_epochs must be greater than 0")
        if self.log_every_n_steps <= 0:
            raise ValueError("log_every_n_steps must be greater than 0")
        if self.gradient_clip_val < 0:
            raise ValueError("gradient_clip_val cannot be negative")


@dataclass(frozen=True)
class Config:
    output_dir: Path = PROJECT_OUTPUT_DIR
    training: TrainingConfig = field(default_factory=TrainingConfig)

    @property
    def checkpoint_dir(self) -> Path:
        return self.output_dir / "checkpoints"

    @property
    def log_dir(self) -> Path:
        return self.output_dir / "logs"

    @property
    def artifact_dir(self) -> Path:
        return self.output_dir / "artifacts"

    @property
    def vocabulary_path(self) -> Path:
        return self.artifact_dir / "vocabulary.json"

    @property
    def word2vec_path(self) -> Path:
        return self.artifact_dir / "word2vec.model"

    def prepare(self) -> None:
        self.training.validate()
        for directory in (self.checkpoint_dir, self.log_dir, self.artifact_dir):
            directory.mkdir(parents=True, exist_ok=True)
