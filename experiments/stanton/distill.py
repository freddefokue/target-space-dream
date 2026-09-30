#!/usr/bin/env python3
"""Distil the ResNet20GN1 teacher into an interpolated-init student under an FE budget.

Phase 1 implements arm A1 (first-order, teacher target at t = 1). Arms A2-A5 are added in
Phase 2. Every run writes ``config.json``, ``metrics.jsonl`` (about every 2% of the budget),
``ckpt.pt`` (every 4%) and ``final.json`` to ``/workspace/runs/<run_id>/`` and resumes from its
last checkpoint when relaunched with the same arguments.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn.utils import parameters_to_vector

from common import (RUNS, ComputeCounter, append_jsonl, budget_fe, git_commit, index_event,
                    ledger_add, load_calibration, save_checkpoint, set_numerics,
                    truncate_metrics, write_json)
from data import CIFAR100GPU, EpochSampler, split_digest
from dream.losses import softmax_kl
from dream.models import ResNet20GN1
from evaluate import evaluate
from teacher import TEACHER_LOGITS, TEACHER_STATE, load_teacher

FIRST_ORDER = {"A1"}
EVALS = 50          # evaluations per run (every 2% of the budget)
CHECKPOINT_EVERY = 2  # checkpoint at every second evaluation (every 4%)


def initial_student(teacher: torch.nn.Module, lam: float, seed: int) -> ResNet20GN1:
    """theta_init = lam * theta_T + (1 - lam) * theta_R(1000 + seed)."""

    torch.manual_seed(1000 + seed)
    student = ResNet20GN1().cuda()
    teacher_state = teacher.state_dict()
    mixed = OrderedDict((name, lam * teacher_state[name] + (1.0 - lam) * value)
                        for name, value in student.state_dict().items())
    student.load_state_dict(mixed)
    return student.train()


def make_optimizer(config: dict, parameters, peak_lr: float) -> torch.optim.Optimizer:
    if config["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, lr=peak_lr, momentum=config["momentum"],
                               nesterov=config["nesterov"], weight_decay=config["weight_decay"])
    if config["optimizer"] == "adam":
        return torch.optim.Adam(parameters, lr=peak_lr, betas=tuple(config["betas"]),
                                weight_decay=config["weight_decay"])
    raise ValueError(f"unknown optimizer {config['optimizer']}")


def run_id_for(config: dict, lam: float, seed: int, budget: str) -> str:
    return f"{config['name']}_lam{lam:g}_s{seed}_{budget}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", default="tune", help="full, tune, or a fraction of B_full")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--runs", type=Path, default=RUNS)
    arguments = parser.parse_args()
    set_numerics()

    config = json.loads(arguments.config.read_text())
    if config["arm"] not in FIRST_ORDER:
        raise NotImplementedError(f"arm {config['arm']} is implemented in Phase 2")
    calibration = load_calibration()
    budget = budget_fe(arguments.budget, calibration)
    run_id = run_id_for(config, arguments.lam, arguments.seed, arguments.budget)
    run_dir = arguments.runs / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path, metrics_path = run_dir / "ckpt.pt", run_dir / "metrics.jsonl"
    if (run_dir / "final.json").exists():
        print(f"{run_id} already finished", flush=True)
        return

    data = CIFAR100GPU()
    teacher = load_teacher()
    teacher_logits = {k: v.cuda() for k, v in torch.load(TEACHER_LOGITS, weights_only=True).items()}
    student = initial_student(teacher, arguments.lam, arguments.seed)
    theta_init = parameters_to_vector(student.parameters()).detach().clone()

    lr_min = float(config["lr_min"])
    peak_lr = max(config["lr"] * (1.0 - arguments.lam), lr_min)  # gnosis (1 - lam) scaling
    optimizer = make_optimizer(config, student.parameters(), peak_lr)
    sampler = EpochSampler(data, config["batch"], seed=arguments.seed)
    counter = ComputeCounter(calibration)
    temperature = float(config.get("temperature", 1.0))

    record = {"run_id": run_id, "config": config, "lam": arguments.lam, "seed": arguments.seed,
              "budget_name": arguments.budget, "budget_fe": budget, "peak_lr": peak_lr,
              "student_random_seed": 1000 + arguments.seed, "data_seed": arguments.seed,
              "teacher_state": str(TEACHER_STATE), "splits": split_digest(), "git": git_commit(),
              "calibration_measured_at": calibration["measured_at"], "B_full": calibration["B_full"],
              "concurrency": arguments.concurrency}
    step, wall_before, next_eval = 0, 0.0, 0
    loss_sum, loss_count = 0.0, 0
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        student.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        sampler.load_state_dict(state["sampler"])
        counter.load_state_dict(state["counter"])
        step, wall_before, next_eval = state["step"], state["wall"], state["next_eval"]
        truncate_metrics(metrics_path, step)
        index_event(run_id, "resume", step=step, fe=counter.fe)
        print(f"{run_id}: resumed at step {step}", flush=True)
    else:
        write_json(run_dir / "config.json", record)
        index_event(run_id, "start", **{k: record[k] for k in ("lam", "seed", "budget_name")},
                    config=config["name"])

    start = time.time()

    def log(done: bool = False) -> dict:
        nonlocal loss_sum, loss_count
        row = {"step": step, "fe": counter.fe, "fe_secondary": counter.fe_secondary,
               "fraction": counter.fe / budget, "wall": wall_before + time.time() - start,
               "lr": optimizer.param_groups[0]["lr"],
               "train_loss": loss_sum / loss_count if loss_count else None,
               "displacement": float(torch.linalg.vector_norm(
                   parameters_to_vector(student.parameters()).detach() - theta_init)),
               "epochs": sampler.epochs_started, "final": done}
        row.update(evaluate(student, data, teacher_logits))
        append_jsonl(metrics_path, row)
        loss_sum, loss_count = 0.0, 0
        return row

    while True:
        if next_eval <= EVALS and counter.fe >= next_eval * budget / EVALS:
            row = log()
            print(f"{run_id} {row['fraction']:.2f} val_agree {row['val_agreement']:.4f} "
                  f"val_kl {row['val_kl']:.4f} train_agree {row['train_subset_agreement']:.4f}",
                  flush=True)
            if not row["val_finite"]:
                raise FloatingPointError(f"{run_id}: non-finite student outputs at step {step}")
            next_eval += 1
            if next_eval % CHECKPOINT_EVERY == 0:
                save_checkpoint(ckpt_path, {
                    "model": student.state_dict(), "optimizer": optimizer.state_dict(),
                    "sampler": sampler.state_dict(), "counter": counter.state_dict(),
                    "step": step, "wall": wall_before + time.time() - start,
                    "next_eval": next_eval})
        images, _ = sampler.next()
        batch = images.shape[0]
        step_cost = counter.cost("fwd_bwd", batch) + counter.cost("fwd", batch)
        if counter.fe + step_cost > budget:
            break
        fraction = counter.fe / budget
        lr = lr_min + 0.5 * (peak_lr - lr_min) * (1.0 + math.cos(math.pi * fraction))
        for group in optimizer.param_groups:
            group["lr"] = lr
        with torch.no_grad():
            target = teacher(images)
        counter.charge("fwd", batch)
        loss = temperature**2 * softmax_kl(target / temperature, student(images) / temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        counter.charge("fwd_bwd", batch)
        optimizer.step()
        step += 1
        loss_sum += float(loss.detach()) if step % 20 == 0 else 0.0
        loss_count += 1 if step % 20 == 0 else 0

    final_row = log(done=True)
    wall_hours = final_row["wall"] / 3600
    total = ledger_add(run_id, wall_hours, arguments.concurrency, "distill")
    final = {**record, **final_row, "wall_hours": wall_hours, "images_by_pass": counter.images,
             "ledger_total_gpu_hours": total}
    write_json(run_dir / "final.json", final)
    index_event(run_id, "end", val_agreement=final_row["val_agreement"],
                val_kl=final_row["val_kl"], fe=counter.fe, wall_hours=wall_hours)
    print(f"{run_id} done: val_agree {final_row['val_agreement']:.4f} "
          f"val_kl {final_row['val_kl']:.4f} ledger {total:.2f} GPU-h", flush=True)


if __name__ == "__main__":
    main()
