#!/usr/bin/env python3
"""Train (or load) the ResNet20GN1 teacher and cache its logits on the fixed evaluation sets.

Recipe (brief §3): Nesterov SGD, momentum 0.9, lr 0.1 cosine to 0 (per step), weight decay 5e-4
on all parameters, batch 128, 200 epochs, crop+flip augmentation, all 50k training images, seed 0.
Resumable from ``ckpt.pt``. The test split is never touched here.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from common import (RUNS, append_jsonl, git_commit, index_event, ledger_add, save_checkpoint,
                    set_numerics, truncate_metrics, write_json)
from data import CIFAR100GPU, EpochSampler, split_digest
from dream.models import ResNet20GN1

TEACHER_DIR = RUNS / "teacher_seed0"
TEACHER_STATE = TEACHER_DIR / "teacher_state.pt"
TEACHER_LOGITS = TEACHER_DIR / "teacher_logits.pt"
PROVIDED = Path("/workspace/inputs/teacher_state.pt")


@torch.no_grad()
def logits_on(model: torch.nn.Module, fixed) -> torch.Tensor:
    model.eval()
    return torch.cat([model(images) for images, _ in fixed.batches(1000)])


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean())


def load_teacher(device: str = "cuda") -> torch.nn.Module:
    model = ResNet20GN1().to(device)
    model.load_state_dict(torch.load(TEACHER_STATE, map_location=device, weights_only=True))
    return model.eval().requires_grad_(False)


def cache_logits(model: torch.nn.Module, data: CIFAR100GPU) -> dict:
    cached = {name: logits_on(model, getattr(data, name)).cpu()
              for name in ("val", "train_subset", "monitor")}
    torch.save(cached, TEACHER_LOGITS)
    return {f"{name}_accuracy": accuracy(cached[name].cuda(), getattr(data, name).labels)
            for name in cached}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=1)
    arguments = parser.parse_args()
    set_numerics()
    data = CIFAR100GPU()
    TEACHER_DIR.mkdir(parents=True, exist_ok=True)

    if PROVIDED.exists():
        TEACHER_STATE.write_bytes(PROVIDED.read_bytes())
        print("loaded provided teacher", cache_logits(load_teacher(), data))
        return

    batch, epochs, peak = 128, arguments.epochs, 0.1
    steps_per_epoch = math.ceil(len(data) / batch)
    total_steps = epochs * steps_per_epoch
    torch.manual_seed(0)
    model = ResNet20GN1().cuda().train()
    optimizer = torch.optim.SGD(model.parameters(), lr=peak, momentum=0.9, nesterov=True,
                                weight_decay=5e-4)
    sampler = EpochSampler(data, batch, seed=0)
    step, wall_before = 0, 0.0
    ckpt_path, metrics_path = TEACHER_DIR / "ckpt.pt", TEACHER_DIR / "metrics.jsonl"
    config = {"model": "ResNet20GN1", "optimizer": "SGD nesterov m0.9", "lr": peak,
              "schedule": "cosine per step to 0", "weight_decay": 5e-4, "batch": batch,
              "epochs": epochs, "seed": 0, "augmentation": "pad4 crop32 + hflip (GPU)",
              "preprocessing": "x/127.5-1", "precision": "float32, TF32 off",
              "splits": split_digest(), "git": git_commit()}
    write_json(TEACHER_DIR / "config.json", config)
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        torch.set_rng_state(state["cpu_rng"].cpu())
        step, wall_before = state["step"], state["wall"]
        truncate_metrics(metrics_path, step)
        print("resumed at step", step, flush=True)
    else:
        index_event("teacher_seed0", "start", config=config)

    start = time.time()
    while step < total_steps:
        lr = 0.5 * peak * (1 + math.cos(math.pi * step / total_steps))
        for group in optimizer.param_groups:
            group["lr"] = lr
        images, labels = sampler.next()
        loss = F.cross_entropy(model(images), labels)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        step += 1
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch
            row = {"step": step, "epoch": epoch, "lr": lr, "loss_sampled": float(loss),
                   "wall": wall_before + time.time() - start}
            if epoch % 10 == 0 or epoch == epochs:
                row["val_accuracy"] = accuracy(logits_on(model, data.val), data.val.labels)
                model.train()
                save_checkpoint(ckpt_path, {"model": model.state_dict(),
                                            "optimizer": optimizer.state_dict(),
                                            "sampler": sampler.state_dict(),
                                            "cpu_rng": torch.get_rng_state(), "step": step,
                                            "wall": wall_before + time.time() - start})
            append_jsonl(metrics_path, row)
            print(row, flush=True)

    torch.save(model.state_dict(), TEACHER_STATE)
    summary = cache_logits(model, data)
    wall = wall_before + time.time() - start
    summary.update({"wall_hours": wall / 3600, "config": config})
    write_json(TEACHER_DIR / "final.json", summary)
    total = ledger_add("teacher_seed0", wall / 3600, arguments.concurrency, "teacher training")
    index_event("teacher_seed0", "end", summary=summary)
    print("teacher done", summary, "ledger total GPU-h", round(total, 3), flush=True)


if __name__ == "__main__":
    main()
