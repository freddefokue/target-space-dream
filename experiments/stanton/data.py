"""GPU-resident CIFAR-100, fixed splits and on-GPU crop+flip augmentation.

Preprocessing follows the thesis: uint8 -> x / 127.5 - 1, no mean/std normalization. Crops pad
the uint8 image with zeros (as torchvision's ``RandomCrop(32, padding=4)`` on PIL images), so the
padding maps to -1 after scaling.

The official test set is split into a validation half (tuning and Phase 1-3 reporting) and a
test half. The test half is only returned when ``open_test=True`` is passed explicitly.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

DATA_ROOT = Path("/workspace/data")
CIFAR_DIR = DATA_ROOT / "cifar-100-python"
SPLIT_FILE = DATA_ROOT / "splits" / "stanton_splits_seed0.pt"
CLASSES = 100


def _load_pickle(name: str) -> tuple[np.ndarray, np.ndarray]:
    with open(CIFAR_DIR / name, "rb") as handle:
        batch = pickle.load(handle, encoding="latin1")
    images = np.asarray(batch["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    labels = np.asarray(batch["fine_labels"], dtype=np.int64)
    return images, labels


def _balanced_pick(labels: np.ndarray, per_class: int, rng: np.random.Generator,
                   exclude: np.ndarray | None = None) -> np.ndarray:
    chosen = []
    for cls in range(CLASSES):
        pool = np.flatnonzero(labels == cls)
        if exclude is not None:
            pool = np.setdiff1d(pool, exclude)
        chosen.append(rng.choice(pool, size=per_class, replace=False))
    return np.sort(np.concatenate(chosen))


def build_splits() -> dict[str, np.ndarray]:
    """Deterministic class-balanced splits from seed 0 (see PREREGISTRATION.md §2)."""

    _, train_labels = _load_pickle("train")
    _, test_labels = _load_pickle("test")
    rng = np.random.default_rng(0)
    val = _balanced_pick(test_labels, 50, rng)
    test = np.setdiff1d(np.arange(len(test_labels)), val)
    train_subset = _balanced_pick(train_labels, 50, rng)
    monitor = _balanced_pick(train_labels, 20, rng, exclude=train_subset)
    return {"val": val, "test": test, "train_subset": train_subset, "monitor": monitor}


def splits() -> dict[str, np.ndarray]:
    """Load the frozen split indices, creating them once on first use."""

    if not SPLIT_FILE.exists():
        SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
        built = build_splits()
        tmp = SPLIT_FILE.with_suffix(".tmp")
        torch.save({k: torch.from_numpy(v) for k, v in built.items()}, tmp)
        tmp.replace(SPLIT_FILE)
    loaded = torch.load(SPLIT_FILE, weights_only=True)
    return {k: v.numpy() for k, v in loaded.items()}


def split_digest() -> str:
    digest = hashlib.sha256()
    for name, indices in sorted(splits().items()):
        digest.update(name.encode())
        digest.update(np.ascontiguousarray(indices).tobytes())
    return digest.hexdigest()[:16]


def to_input(images_u8: torch.Tensor) -> torch.Tensor:
    return images_u8.float() / 127.5 - 1.0


@dataclass
class FixedSet:
    """A fixed, unaugmented image set held on the GPU."""

    images_u8: torch.Tensor
    labels: torch.Tensor

    def __len__(self) -> int:
        return self.labels.numel()

    def batches(self, size: int = 1000):
        for start in range(0, len(self), size):
            yield to_input(self.images_u8[start : start + size]), self.labels[start : start + size]


class CIFAR100GPU:
    """The 50k training images on the GPU plus the fixed evaluation sets."""

    def __init__(self, device: str | torch.device = "cuda", *, open_test: bool = False) -> None:
        self.device = torch.device(device)
        train_images, train_labels = _load_pickle("train")
        test_images, test_labels = _load_pickle("test")
        index = splits()
        self.train_u8 = torch.from_numpy(train_images).to(self.device)
        self.train_labels = torch.from_numpy(train_labels).to(self.device)

        def fixed(images: np.ndarray, labels: np.ndarray, rows: np.ndarray) -> FixedSet:
            return FixedSet(
                torch.from_numpy(images[rows]).to(self.device),
                torch.from_numpy(labels[rows]).to(self.device),
            )

        self.val = fixed(test_images, test_labels, index["val"])
        self.train_subset = fixed(train_images, train_labels, index["train_subset"])
        self.monitor = fixed(train_images, train_labels, index["monitor"])
        self.test = fixed(test_images, test_labels, index["test"]) if open_test else None

    def __len__(self) -> int:
        return self.train_labels.numel()


def augment(images_u8: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Zero-pad 4, random 32x32 crop and random horizontal flip, as one gather on the GPU."""

    count = images_u8.shape[0]
    device = images_u8.device
    padded = torch.nn.functional.pad(images_u8, (4, 4, 4, 4))
    offsets = torch.randint(0, 9, (2, count), device=device, generator=generator)
    flip = torch.rand(count, device=device, generator=generator) < 0.5
    ramp = torch.arange(32, device=device)
    rows = offsets[0][:, None] + ramp
    columns = torch.where(flip[:, None], 31 - ramp, ramp) + offsets[1][:, None]
    batch = torch.arange(count, device=device)[:, None, None]
    crops = padded.permute(0, 2, 3, 1)[batch, rows[:, :, None], columns[:, None, :]]
    return crops.permute(0, 3, 1, 2)


class EpochSampler:
    """Seeded shuffled mini-batches with an explicit, checkpointable generator state."""

    def __init__(self, dataset: CIFAR100GPU, batch_size: int, seed: int) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.generator = torch.Generator(device=dataset.device)
        self.generator.manual_seed(seed)
        self._order: torch.Tensor | None = None
        self._cursor = 0
        self.epochs_started = 0

    def next(self) -> tuple[torch.Tensor, torch.Tensor]:
        size = len(self.dataset)
        if self._order is None or self._cursor >= size:
            self._order = torch.randperm(size, device=self.dataset.device, generator=self.generator)
            self._cursor = 0
            self.epochs_started += 1
        rows = self._order[self._cursor : self._cursor + self.batch_size]
        self._cursor += rows.numel()
        images = augment(self.dataset.train_u8[rows], self.generator)
        return to_input(images), self.dataset.train_labels[rows]

    def state_dict(self) -> dict:
        return {
            "generator": self.generator.get_state(),
            "order": self._order,
            "cursor": self._cursor,
            "epochs_started": self.epochs_started,
        }

    def load_state_dict(self, state: dict) -> None:
        self.generator.set_state(state["generator"].cpu())
        self._order = state["order"]
        self._cursor = state["cursor"]
        self.epochs_started = state["epochs_started"]
