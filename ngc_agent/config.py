"""Configuration validation without importing a training or API dependency."""
from dataclasses import asdict, dataclass, fields
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TrainConfig:
    train_csv: str = "splits1/train.csv"
    val_csv: str = "splits1/val.csv"
    test_csv: str = "splits1/test.csv"
    episodes: int = 500
    eval_every: int = 50
    eval_episodes: int = 100
    val_random_eval_episodes: int = 100
    test_episodes: int = 200
    k_shot: int = 20
    q_shot: int = 20
    embed_dim: int = 128
    dropout: float = 0.1
    temperature: float = 10.0
    lr: float = 0.001
    weight_decay: float = 0.0001
    seed: int = 2026
    device: str = ""
    no_prior: bool = False
    auto_threshold: bool = True
    timeout_seconds: int = 14400

    def __post_init__(self):
        for name, lo, hi in (
            ("episodes", 1, 100000), ("eval_every", 1, 100000),
            ("eval_episodes", 1, 10000), ("val_random_eval_episodes", 1, 10000),
            ("test_episodes", 1, 10000), ("k_shot", 1, 1000),
            ("q_shot", 1, 1000), ("embed_dim", 8, 4096),
            ("seed", 0, 2**32 - 1), ("timeout_seconds", 60, 604800),
        ):
            value = getattr(self, name)
            if type(value) is not int or not lo <= value <= hi:
                raise ValueError(f"{name} must be an integer in [{lo}, {hi}]")
        for name, lo, hi in (
            ("lr", 1e-6, 0.1), ("dropout", 0.0, 0.9),
            ("temperature", 0.01, 100.0), ("weight_decay", 0.0, 1.0),
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not lo <= value <= hi:
                raise ValueError(f"{name} must be finite and in [{lo}, {hi}]")
        for name in ("train_csv", "val_csv", "test_csv"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} must be a nonempty path")
        for name in ("no_prior", "auto_threshold"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.device, str) or (self.device and self.device != "cpu" and self.device != "cuda" and not (self.device.startswith("cuda:") and self.device[5:].isdigit())):
            raise ValueError("device must be empty, cpu, cuda or cuda:N")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("Training configuration must be a JSON object")
        unknown = set(value) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
        return cls(**value)

    def to_dict(self):
        return asdict(self)

    def command_args(self, root):
        result = []
        for name, value in self.to_dict().items():
            if name == "timeout_seconds":
                continue
            if isinstance(value, bool):
                if value:
                    result.append(f"--{name}")
            elif value != "":
                if name.endswith("_csv"):
                    path = Path(value).expanduser()
                    value = str(path if path.is_absolute() else root / path)
                result.extend([f"--{name}", str(value)])
        # Use reproducible generated episode banks; a local bank for another k-shot
        # configuration must not silently change a planned experiment.
        return result + ["--val_fixed_bank", "", "--test_fixed_bank", "", "--disable_tqdm"]
