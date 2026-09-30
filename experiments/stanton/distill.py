#!/usr/bin/env python3
"""Distil the ResNet20GN1 teacher into an interpolated-init student under an FE budget.

Arms (PREREGISTRATION.md §6):

- A1 first-order, teacher target (optionally Stanton's tau^2-scaled tempered loss, warmup);
- A2 first-order, moving target ``y_t = (1 - t) f_init + t f_T`` (fixed or adaptive t);
- A3 damped GGN-CG with Levenberg-Marquardt damping, teacher target;
- A4 soft Dream: A3's step on the moving target;
- A5 first-order, tempered teacher (geometric T schedule) or Annealing-KD (MSE on Phi(T) f_T).

Every run writes ``config.json``, ``metrics.jsonl`` (about every 2% of the budget), ``ckpt.pt``
(every 4%), ``final_model.pt`` and ``final.json`` to ``/workspace/runs/<run_id>/`` and resumes from
its last checkpoint when relaunched with the same arguments. ``--init-from`` starts from another
run's final weights (hand-off diagnostic).
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.func import functional_call
from torch.nn import functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from common import (RUNS, ComputeCounter, append_jsonl, budget_fe, git_commit, index_event,
                    ledger_add, load_calibration, save_checkpoint, set_numerics,
                    truncate_metrics, write_json)
from data import CIFAR100GPU, EpochSampler, split_digest
from dream.ggn import (GaussNewtonOperator, JacobianProducts, LevenbergMarquardt, damped_cg,
                       reduction_ratio, softmax_output_metric)
from dream.losses import softmax_kl
from dream.models import ResNet20GN1
from dream.parameters import ParameterSpec
from dream.schedules import (AdaptiveT, annealing_kd_factor, annealing_kd_temperature, fixed_t,
                             geometric_temperature, warmup_cosine)
from evaluate import evaluate
from teacher import TEACHER_LOGITS, TEACHER_STATE, load_teacher

ARMS = {"A1", "A2", "A3", "A4", "A5"}
FIRST_ORDER = {"A1", "A2", "A5"}
MOVING = {"A2", "A4"}
EVALS = 50            # evaluations per run (every 2% of the budget)
CHECKPOINT_EVERY = 2  # checkpoint at every second evaluation (every 4%)
MONITOR_CHECKS = 200  # adaptive-t checks per run (every 0.5% of the budget)


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


def run_id_for(config: dict, lam: float, seed: int, budget: str, init_from: Path | None) -> str:
    run_id = f"{config['name']}_lam{lam:g}_s{seed}_{budget}"
    return run_id + (f"_from-{init_from.name}" if init_from else "")


class Schedule:
    """t (A2/A4) or temperature (A5) as a function of the budget fraction."""

    def __init__(self, config: dict) -> None:
        self.config = config.get("schedule")
        self.adaptive = (AdaptiveT(threshold=self.config["threshold"])
                         if self.config and self.config["kind"] == "adaptive" else None)

    def t(self, s: float) -> float:
        if self.adaptive is not None:
            return self.adaptive.value(s)
        return fixed_t(s, self.config["phi"])

    def temperature(self, s: float) -> float:
        if self.config["kind"] == "geometric":
            return geometric_temperature(s, self.config["T0"], self.config["phi"])
        return float(annealing_kd_temperature(s, self.config["tau_max"], self.config["phi"]))

    def state_dict(self) -> dict:
        return self.adaptive.state_dict() if self.adaptive else {}

    def load_state_dict(self, state: dict) -> None:
        if self.adaptive:
            self.adaptive.load_state_dict(state)


class Distiller:
    """Shared state and the target construction of all arms."""

    def __init__(self, config: dict, arguments, data: CIFAR100GPU, teacher: torch.nn.Module,
                 student: torch.nn.Module, counter: ComputeCounter) -> None:
        self.config, self.arm = config, config["arm"]
        self.lam = arguments.lam
        self.data, self.teacher, self.student, self.counter = data, teacher, student, counter
        self.schedule = Schedule(config)
        self.init_model = None
        if self.arm in MOVING:
            self.init_model = copy.deepcopy(student).eval().requires_grad_(False)
        self.stats: dict[str, float] = {}

    def _stat(self, key: str, value: float) -> None:
        total, count = self.stats.get(key, (0.0, 0))
        self.stats[key] = (total + value, count + 1)

    def pop_stats(self) -> dict:
        averaged = {f"mean_{key}": total / count for key, (total, count) in self.stats.items()}
        self.stats = {}
        return averaged

    def target(self, images: torch.Tensor, s: float) -> tuple[torch.Tensor, dict]:
        """Charged target logits for a training batch and the schedule values used."""

        batch = images.shape[0]
        with torch.no_grad():
            teacher_logits = self.teacher(images)
        self.counter.charge("fwd", batch)
        if self.arm in MOVING:
            t = self.schedule.t(s)
            with torch.no_grad():
                initial_logits = self.init_model(images)
            self.counter.charge("fwd", batch)
            return torch.lerp(initial_logits, teacher_logits, t), {"t": t}
        if self.arm == "A5":
            temperature = self.schedule.temperature(s)
            if self.config["schedule"]["kind"] == "annealing_kd":
                factor = annealing_kd_factor(temperature, self.config["schedule"]["tau_max"])
                return factor * teacher_logits, {"T": temperature, "phi_T": factor}
            return teacher_logits / temperature, {"T": temperature}
        return teacher_logits, {}

    def loss(self, target: torch.Tensor, student_logits: torch.Tensor) -> torch.Tensor:
        if self.arm == "A5" and self.config["schedule"]["kind"] == "annealing_kd":
            return F.mse_loss(student_logits, target)
        if self.arm == "A1":
            tau = float(self.config.get("temperature", 1.0))
            return tau**2 * softmax_kl(target / tau, student_logits / tau)
        return softmax_kl(target, student_logits)

    # Adaptive t: monitor checks -------------------------------------------------------------

    def prepare_monitor(self, teacher_monitor: torch.Tensor) -> None:
        if self.schedule.adaptive is None:
            return
        self.monitor_teacher = teacher_monitor
        with torch.no_grad():
            self.monitor_initial = torch.cat(
                [self.init_model(images) for images, _ in self.data.monitor.batches(1000)])

    def charge_monitor_setup(self) -> None:
        """Teacher and f_init logits on the fixed monitor set are charged once per run."""

        if self.schedule.adaptive is not None:
            self.counter.charge("fwd", len(self.data.monitor), times=2)

    def monitor_check(self, s: float) -> dict | None:
        control = self.schedule.adaptive
        t = control.value(s)
        if not control.controlling:
            return None
        was_training = self.student.training
        self.student.eval()
        with torch.no_grad():
            student_logits = torch.cat(
                [self.student(images) for images, _ in self.data.monitor.batches(1000)])
        self.student.train(was_training)
        self.counter.charge("fwd", len(self.data.monitor))
        kl = float(softmax_kl(torch.lerp(self.monitor_initial, self.monitor_teacher, t),
                              student_logits))
        return control.check(kl, s) or {"reason": None, "kl": kl, "t": t}


class FirstOrder(Distiller):
    def __init__(self, config, arguments, data, teacher, student, counter) -> None:
        super().__init__(config, arguments, data, teacher, student, counter)
        self.lr_min = float(config["lr_min"])
        self.peak_lr = max(config["lr"] * (1.0 - arguments.lam), self.lr_min)  # gnosis scaling
        self.warmup = float(config.get("warmup", 0.0))
        self.optimizer = make_optimizer(config, student.parameters(), self.peak_lr)
        self.batch = config["batch"]

    def step_cost(self, batch: int) -> float:
        passes = 2 if self.arm in MOVING else 1
        return self.counter.cost("fwd_bwd", batch) + passes * self.counter.cost("fwd", batch)

    def step(self, images: torch.Tensor, s: float) -> dict:
        lr = warmup_cosine(s, self.peak_lr, self.lr_min, self.warmup)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        target, values = self.target(images, s)
        loss = self.loss(target, self.student(images))
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.counter.charge("fwd_bwd", images.shape[0])
        self.optimizer.step()
        return {"loss": loss, "lr": lr, **values}

    def state_dict(self) -> dict:
        return {"optimizer": self.optimizer.state_dict(), "schedule": self.schedule.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.optimizer.load_state_dict(state["optimizer"])
        self.schedule.load_state_dict(state["schedule"])


class GaussNewton(Distiller):
    """Damped GGN-CG with Levenberg-Marquardt control (A3, A4)."""

    def __init__(self, config, arguments, data, teacher, student, counter,
                 method: str) -> None:
        super().__init__(config, arguments, data, teacher, student, counter)
        settings = config["gn"]
        self.batch = settings["batch"]
        self.curvature_batch = settings["curvature_batch"]
        self.iterations = settings["cg_iterations"]
        self.beta = settings["warm_start"]
        self.method = method
        self.lm = LevenbergMarquardt(settings["damping"])
        params = OrderedDict((n, p.detach()) for n, p in student.named_parameters())
        self.spec = ParameterSpec.from_params(params)
        self.theta = self.spec.flatten(params).clone()
        self.previous = torch.zeros_like(self.theta)

    def _function(self, images: torch.Tensor):
        def function(flat: torch.Tensor) -> torch.Tensor:
            return functional_call(self.student, self.spec.unflatten(flat), (images,))
        return function

    def step_cost(self, batch: int) -> float:
        """Upper bound of one step (warm start plus K products), used for the budget stop."""

        curvature = min(self.curvature_batch, batch)
        passes = 2 if self.arm in MOVING else 1
        return (self.counter.cost("fwd_bwd", batch)
                + (passes + 1) * self.counter.cost("fwd", batch)
                + self.counter.cost(f"ggn_setup_{self.method}", curvature)
                + (self.iterations + 1) * self.counter.cost(f"ggn_product_{self.method}",
                                                            curvature))

    def step(self, images: torch.Tensor, s: float) -> dict:
        batch = images.shape[0]
        target, values = self.target(images, s)
        function = self._function(images)
        with torch.enable_grad():
            leaf = self.theta.clone().requires_grad_(True)
            loss = softmax_kl(target, function(leaf))
            (gradient,) = torch.autograd.grad(loss, leaf)
        self.counter.charge("fwd_bwd", batch)
        loss_value = float(loss)

        curvature_images = images[: self.curvature_batch]
        size = curvature_images.shape[0]
        products = JacobianProducts(self._function(curvature_images), self.theta, self.method)
        self.counter.charge(f"ggn_setup_{self.method}", size)
        operator = GaussNewtonOperator(products, softmax_output_metric(products.output))
        damping = self.lm.damping
        solve = damped_cg(operator, -gradient, damping, iterations=self.iterations,
                          initial=self.beta * self.previous)
        self.counter.charge(f"ggn_product_{self.method}", size, times=operator.products_applied)
        delta = solve.solution
        step_curvature = float(torch.dot(delta, -gradient - solve.residual - damping * delta))
        gradient_dot = float(torch.dot(gradient, delta))
        del products, operator

        with torch.no_grad():
            trial = float(softmax_kl(target, function(self.theta + delta)))
        self.counter.charge("fwd", batch)
        finite = math.isfinite(trial) and bool(torch.isfinite(delta).all())
        if finite:
            ratio = reduction_ratio(trial - loss_value, gradient_dot, step_curvature)
            self.lm.update(ratio)
            accepted = trial <= loss_value
            self.previous = delta
        else:
            ratio, accepted = float("nan"), False
            self.lm.damping = min(self.lm.damping * 10.0, self.lm.maximum)
            self.previous = torch.zeros_like(self.theta)
        if accepted:
            self.theta = self.theta + delta
            vector_to_parameters(self.theta, self.student.parameters())
        self._stat("accepted", float(accepted))
        self._stat("damping", damping)
        self._stat("rho", ratio if math.isfinite(ratio) else -10.0)
        self._stat("cg_products", solve.products)
        self._stat("cg_relative_residual", solve.relative_residual)
        self._stat("step_norm", float(torch.linalg.vector_norm(delta)))
        self._stat("nonfinite", float(not finite))
        return {"loss": loss_value, **values}

    def state_dict(self) -> dict:
        return {"theta": self.theta, "previous": self.previous, "damping": self.lm.damping,
                "schedule": self.schedule.state_dict()}

    def load_state_dict(self, state: dict) -> None:
        self.theta = state["theta"]
        self.previous = state["previous"]
        self.lm.damping = state["damping"]
        self.schedule.load_state_dict(state["schedule"])
        vector_to_parameters(self.theta, self.student.parameters())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lam", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--budget", default="tune", help="full, tune, or a fraction of B_full")
    parser.add_argument("--init-from", type=Path, default=None,
                        help="run directory whose final_model.pt is the starting point")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--runs", type=Path, default=RUNS)
    arguments = parser.parse_args()
    set_numerics()

    config = json.loads(arguments.config.read_text())
    if config["arm"] not in ARMS:
        raise ValueError(f"unknown arm {config['arm']}")
    calibration = load_calibration()
    budget = budget_fe(arguments.budget, calibration)
    run_id = run_id_for(config, arguments.lam, arguments.seed, arguments.budget,
                        arguments.init_from)
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
    if arguments.init_from is not None:
        student.load_state_dict(torch.load(arguments.init_from / "final_model.pt",
                                           map_location="cuda", weights_only=True))
    theta_init = parameters_to_vector(student.parameters()).detach().clone()
    theta_teacher = parameters_to_vector(teacher.parameters()).detach().clone()
    counter = ComputeCounter(calibration)
    if config["arm"] in FIRST_ORDER:
        arm = FirstOrder(config, arguments, data, teacher, student, counter)
    else:
        arm = GaussNewton(config, arguments, data, teacher, student, counter,
                          calibration["ggn_method"])
    arm.prepare_monitor(teacher_logits["monitor"])
    sampler = EpochSampler(data, arm.batch, seed=arguments.seed)

    record = {"run_id": run_id, "config": config, "lam": arguments.lam, "seed": arguments.seed,
              "budget_name": arguments.budget, "budget_fe": budget,
              "peak_lr": getattr(arm, "peak_lr", None),
              "ggn_method": getattr(arm, "method", None),
              "init_from": str(arguments.init_from) if arguments.init_from else None,
              "student_random_seed": 1000 + arguments.seed, "data_seed": arguments.seed,
              "teacher_state": str(TEACHER_STATE), "splits": split_digest(), "git": git_commit(),
              "calibration_measured_at": calibration["measured_at"],
              "B_full": calibration["B_full"], "concurrency": arguments.concurrency}
    step, wall_before, next_eval, next_check = 0, 0.0, 0, 0
    loss_sum, loss_count = 0.0, 0
    last_values: dict = {}
    if ckpt_path.exists():
        state = torch.load(ckpt_path, map_location="cuda", weights_only=False)
        student.load_state_dict(state["model"])
        arm.load_state_dict(state["arm"])
        sampler.load_state_dict(state["sampler"])
        counter.load_state_dict(state["counter"])
        step, wall_before = state["step"], state["wall"]
        next_eval, next_check = state["next_eval"], state["next_check"]
        truncate_metrics(metrics_path, step)
        index_event(run_id, "resume", step=step, fe=counter.fe)
        print(f"{run_id}: resumed at step {step}", flush=True)
    else:
        write_json(run_dir / "config.json", record)
        index_event(run_id, "start", **{k: record[k] for k in ("lam", "seed", "budget_name")},
                    config=config["name"], init_from=record["init_from"])
        arm.charge_monitor_setup()

    start = time.time()

    def log(done: bool = False) -> dict:
        nonlocal loss_sum, loss_count
        theta = parameters_to_vector(student.parameters()).detach()
        row = {"step": step, "fe": counter.fe, "fe_secondary": counter.fe_secondary,
               "fraction": counter.fe / budget, "wall": wall_before + time.time() - start,
               "train_loss": loss_sum / loss_count if loss_count else None,
               "displacement": float(torch.linalg.vector_norm(theta - theta_init)),
               "teacher_distance": float(torch.linalg.vector_norm(theta - theta_teacher)),
               "epochs": sampler.epochs_started, "final": done,
               **{k: v for k, v in last_values.items() if k != "loss"}, **arm.pop_stats()}
        if arm.schedule.adaptive is not None:
            row["t"] = arm.schedule.adaptive.t
        row.update(evaluate(student, data, teacher_logits))
        append_jsonl(metrics_path, row)
        loss_sum, loss_count = 0.0, 0
        return row

    def checkpoint() -> None:
        save_checkpoint(ckpt_path, {
            "model": student.state_dict(), "arm": arm.state_dict(),
            "sampler": sampler.state_dict(), "counter": counter.state_dict(), "step": step,
            "wall": wall_before + time.time() - start, "next_eval": next_eval,
            "next_check": next_check})

    while True:
        if arm.schedule.adaptive is not None and counter.fe >= next_check * budget / MONITOR_CHECKS:
            event = arm.monitor_check(counter.fe / budget)
            next_check += 1
            if event is not None and event.get("reason"):
                append_jsonl(run_dir / "schedule_events.jsonl", {"step": step, **event})
        if next_eval <= EVALS and counter.fe >= next_eval * budget / EVALS:
            row = log()
            print(f"{run_id} {row['fraction']:.2f} val_agree {row['val_agreement']:.4f} "
                  f"val_kl {row['val_kl']:.4f} train_agree {row['train_subset_agreement']:.4f}"
                  + (f" t {row['t']:.3f}" if "t" in row else ""), flush=True)
            if not row["val_finite"]:
                raise FloatingPointError(f"{run_id}: non-finite student outputs at step {step}")
            next_eval += 1
            if next_eval % CHECKPOINT_EVERY == 0:
                checkpoint()
        images, _ = sampler.next()
        if counter.fe + arm.step_cost(images.shape[0]) > budget:
            break
        last_values = arm.step(images, counter.fe / budget)
        step += 1
        if step % 20 == 0 or not config["arm"] in FIRST_ORDER:
            loss_sum += float(last_values["loss"])
            loss_count += 1

    final_row = log(done=True)
    torch.save(student.state_dict(), run_dir / "final_model.pt")
    if arm.schedule.adaptive is not None:
        events = arm.schedule.adaptive.events
        final_row["schedule_summary"] = {
            "final_t": arm.schedule.adaptive.t,
            "advances": len([e for e in events if e["reason"] != "deadline"]),
            "forced_relative": len([e for e in events if e["reason"] == "relative"]),
            "deadline_triggered": any(e["reason"] == "deadline" for e in events)}
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
