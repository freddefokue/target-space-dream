#!/usr/bin/env python3
"""Evaluate final students on all 50,000 distillation images without augmentation.

For each run directory: top-1 agreement with the teacher, mean ``KL(p_T || p_S)`` at temperature
1 (the distillation loss on clean images; the quantity Stanton et al. Figure 6(b) calls the train
loss) and label accuracy, written to ``<run>/train_full_eval.json``. Teacher logits on the 50k
images are cached once. Evaluation only: no training, no test split. The GPU time is charged to
the budget ledger.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from common import ledger_add, set_numerics, write_json
from data import CIFAR100GPU, to_input
from dream.models import ResNet20GN1
from teacher import TEACHER_DIR, load_teacher

TRAIN_LOGITS = TEACHER_DIR / "teacher_logits_train50k.pt"
BATCH = 500


@torch.no_grad()
def logits_on_train(model: torch.nn.Module, data: CIFAR100GPU) -> torch.Tensor:
    model.eval()
    return torch.cat([model(to_input(data.train_u8[start:start + BATCH]))
                      for start in range(0, len(data), BATCH)])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", type=Path, nargs="+")
    parser.add_argument("--force", action="store_true")
    arguments = parser.parse_args()
    set_numerics()
    start = time.time()
    data = CIFAR100GPU()
    if TRAIN_LOGITS.exists():
        teacher_logits = torch.load(TRAIN_LOGITS, map_location="cuda", weights_only=True)
    else:
        teacher_logits = logits_on_train(load_teacher(), data)
        torch.save(teacher_logits.cpu(), TRAIN_LOGITS)
    teacher_log_probs = F.log_softmax(teacher_logits, dim=-1)
    student = ResNet20GN1().cuda()
    evaluated = 0
    for run in arguments.runs:
        out = run / "train_full_eval.json"
        if out.exists() and not arguments.force:
            continue
        student.load_state_dict(torch.load(run / "final_model.pt", map_location="cuda",
                                           weights_only=True))
        logits = logits_on_train(student, data)
        kl = (teacher_log_probs.exp() * (teacher_log_probs - F.log_softmax(logits, -1))).sum(-1)
        result = {
            "images": len(data), "augmentation": "none",
            "train_agreement": float((logits.argmax(-1) == teacher_logits.argmax(-1)).float().mean()),
            "train_kl": float(kl.mean()),
            "train_accuracy": float((logits.argmax(-1) == data.train_labels).float().mean()),
            "finite": bool(torch.isfinite(logits).all()),
        }
        write_json(out, result)
        evaluated += 1
        print(run.name, {k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()},
              flush=True)
    torch.cuda.synchronize()
    hours = (time.time() - start) / 3600
    total = ledger_add("train_eval", hours, 1, f"50k clean-train evaluation of {evaluated} run(s)")
    print(f"evaluated {evaluated} run(s) in {hours * 3600:.0f} s; ledger {total:.2f} GPU-h")


if __name__ == "__main__":
    main()
