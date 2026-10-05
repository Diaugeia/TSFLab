"""tsflab.experiments: configs, trainer, objectives, checkpoints, execution, queues, and tracking (fake models only)."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from pydantic import ValidationError
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from tsflab.catalog.registry.losses import LOSS_NAME_MAP, get_loss
from tsflab.data.schemas.datasets.custom import DatasetParameterConfig
from tsflab.experiments.config.schema.evaluation import EvaluationConfig
from tsflab.experiments.config.schema.runtime import ExperimentRuntimeConfig
from tsflab.experiments.config.schema.training import TrainConfig
from tsflab.experiments.infra.api import (
    Budget,
    FileCancellation,
    Storage,
    Tracker,
    UsageLedger,
    any_cancelled,
    describe_modules,
    storage_status,
)
from tsflab.experiments.infra.comparison import compare_rows
from tsflab.experiments.infra.policy import ExecutionPolicy
from tsflab.experiments.runner.callbacks import Callback
from tsflab.experiments.runner.objective import TrainingBatch
from tsflab.experiments.runner.trainer import _forward_training, train
from tsflab.experiments.utils.record import write_run_record

# ---------------------------------------------------------------------------
# Config schemas, run records, and profiling
# ---------------------------------------------------------------------------


class _FourInputModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[tuple[object, ...]] = []

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        self.calls.append((x_enc, x_mark_enc, x_dec, x_mark_dec))
        return x_enc[:, -2:, :]


class ArchitectureBoundaryTests(unittest.TestCase):

    def test_retired_runtime_and_loss_aliases_are_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ExperimentRuntimeConfig.model_validate({"gpus": [0, 1]})
        self.assertNotIn("l1", LOSS_NAME_MAP)

    def test_experiment_and_dataset_schemas_reject_unknown_options(self) -> None:
        with self.assertRaises(ValidationError):
            EvaluationConfig.model_validate({"profiling": True})
        with self.assertRaises(ValidationError):
            TrainConfig.model_validate({"epochs": 1, "learning_rate": 0.01})
        with self.assertRaises(ValidationError):
            DatasetParameterConfig.model_validate(
                {"target": "OT", "normalise_each_channel": True}
            )

    def test_invalid_run_record_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "records" / "invalid.json"
            with (
                patch(
                    "tsflab.experiments.utils.record.build_record_dict",
                    side_effect=ValueError("invalid record"),
                ),
                self.assertRaisesRegex(ValueError, "invalid record"),
            ):
                write_run_record(str(target))
            self.assertFalse(target.exists())

    def test_profiler_uses_canonical_four_input_call_and_restores_mode(self) -> None:
        profile = importlib.import_module("tsflab.experiments.evaluation.profile")
        model = _FourInputModel().eval()
        loader = [
            (
                torch.randn(2, 4, 3),
                torch.randn(2, 2, 3),
                torch.randn(2, 4, 2),
                torch.randn(2, 2, 2),
            )
        ]
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "profile.txt"
            with (
                patch.object(profile, "_try_torchinfo_summary", return_value="summary"),
                patch.object(profile, "_try_flops", return_value="flops"),
                patch.object(profile, "_latency_benchmark", return_value=["latency"]),
            ):
                profile.profile_model(
                    model, loader, torch.device("cpu"), 0, 2, str(target)
                )
            self.assertTrue(target.is_file())
            self.assertIn("summary", target.read_text(encoding="utf-8"))
        self.assertFalse(model.training)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual(len(model.calls[0]), 4)


# ---------------------------------------------------------------------------
# Training objectives (runner contract)
# ---------------------------------------------------------------------------


class _Linear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Linear(6, 3)
        self.aux_loss = None

    def forward(self, x, x_mark=None, dec=None, y_mark=None):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


def _batches(count=2, seq=6, pred=3, channels=2):
    torch.manual_seed(0)
    return [
        (torch.randn(4, seq, channels), torch.randn(4, pred, channels), None, None)
        for _ in range(count)
    ]


def _step(model, objective, x, y, criterion, features="M"):
    return _forward_training(
        model, objective, x, None, torch.zeros_like(y), None, y, y.shape[1], features, criterion
    )


class RunnerContractTests(unittest.TestCase):
    def test_default_path_is_criterion_plus_aux_loss(self) -> None:
        model, criterion = _Linear(), get_loss("mse")
        x, y = torch.randn(4, 6, 2), torch.randn(4, 3, 2)
        outputs, loss = _step(model, None, x, y, criterion)
        self.assertTrue(torch.equal(loss, criterion(outputs, y)))
        model.aux_loss = torch.tensor(0.5)
        _, with_aux = _step(model, None, x, y, criterion)
        self.assertTrue(torch.allclose(with_aux, loss + 0.5))

    def test_objective_receives_batch_and_criterion(self) -> None:
        seen = {}

        def objective(model, batch, criterion):
            seen.update(batch=batch, criterion=criterion)
            forecast = batch.forecast(model)
            return forecast, criterion(batch.align(forecast), batch.target) * 2

        model, criterion = _Linear(), get_loss("mse")
        x, y = torch.randn(4, 6, 2), torch.randn(4, 3, 2)
        outputs, loss = _step(model, objective, x, y, criterion, "MS")
        self.assertIsInstance(seen["batch"], TrainingBatch)
        self.assertIs(seen["criterion"], criterion)
        self.assertEqual(tuple(seen["batch"].target.shape), (4, 3, 1))
        self.assertTrue(torch.isfinite(loss))

    def test_objective_may_return_no_forecast_but_loss_must_be_finite(self) -> None:
        model, criterion = _Linear(), get_loss("mse")
        x, y = torch.randn(4, 6, 2), torch.randn(4, 3, 2)
        outputs, loss = _step(model, lambda m, b, c: (None, b.forecast(m).mean()), x, y, criterion)
        self.assertIsNone(outputs)
        with self.assertRaises(ValueError):
            _step(model, lambda m, b, c: (None, torch.tensor(float("nan"))), x, y, criterion)
        with self.assertRaises(ValueError):
            _step(model, lambda m, b, c: (torch.zeros(4, 3, 5), b.forecast(m).mean()), x, y, criterion)

    def _train(self, objective=None, setup=None, callbacks=None):
        model = _Linear()
        loader = _batches()
        with tempfile.TemporaryDirectory() as tmp:
            train(
                model, loader, loader, torch.device("cpu"), 2, 5, "mse", {},
                torch.optim.SGD(model.parameters(), lr=0.01), "type1", 0.01, 2, 0, 3, "M",
                False, tmp, SimpleNamespace(strategy="best", save_k=1),
                callbacks=callbacks, training_objective=objective, training_setup=setup,
            )
        return model

    def test_train_runs_setup_once_and_supports_missing_forecast_with_callbacks(self) -> None:
        calls = []

        def setup(model, loader, *, pred_len, features):
            calls.append((pred_len, features, len(loader)))

        def objective(model, batch, criterion):
            return None, criterion(batch.align(batch.forecast(model)), batch.target)

        seen = []

        class Spy(Callback):
            def on_compute_loss(self, ctx):
                seen.append(ctx.outputs.requires_grad)

        self._train(objective, setup, [Spy()])
        self.assertEqual(calls, [(3, "M", 2)])
        self.assertTrue(seen and not any(seen))
        self._train(objective, setup)

    def test_validation_ignores_the_objective(self) -> None:
        def objective(model, batch, criterion):
            return None, batch.forecast(model).sum() * 0.0 + 1000.0

        # Validation uses forward + criterion; training objective only affects the loop loss.
        self._train(objective)


# ---------------------------------------------------------------------------
# Training resume, execution policy, comparison, and tracking
# ---------------------------------------------------------------------------


class Forecaster(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = torch.nn.Dropout(0.3)
        self.linear = torch.nn.Linear(4, 2)

    def forward(self, x, marks, decoder, decoder_marks):
        return self.linear(self.dropout(x.transpose(1, 2))).transpose(1, 2)


class InterruptAfterEpoch:
    def start(self, step):
        pass

    def log(self, metrics, step):
        if step == 1:
            raise RuntimeError("simulated interruption after checkpoint commit")


def training_case(directory, *, resume=False, tracker=None):
    torch.set_num_threads(1)
    torch.manual_seed(431)
    model = Forecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    dataset = TensorDataset(
        torch.arange(64).reshape(16, 4, 1).float() / 64,
        torch.arange(32).reshape(16, 2, 1).float() / 32,
        torch.zeros(16, 4, 1),
        torch.zeros(16, 2, 1),
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    result = train(
        model,
        loader,
        loader,
        torch.device("cpu"),
        epochs=3,
        patience=10,
        loss_name="mse",
        loss_params={},
        optimizer=optimizer,
        lradj="exponential",
        base_lr=0.01,
        total_epochs=3,
        label_len=0,
        pred_len=2,
        features="M",
        use_amp=False,
        checkpoint_dir=str(directory),
        checkpoint_cfg=SimpleNamespace(strategy="best", save_k=1),
        resume=resume,
        tracker=tracker,
    )
    return model, optimizer, result


def test_resume_matches_uninterrupted_training_including_rng_and_optimizer(tmp_path):
    expected_model, expected_optimizer, _ = training_case(tmp_path / "full")
    with pytest.raises(RuntimeError, match="simulated"):
        training_case(tmp_path / "resumed", tracker=InterruptAfterEpoch())
    model, optimizer, _ = training_case(tmp_path / "resumed", resume=True)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(
            value, expected_model.state_dict()[key], rtol=0, atol=0
        )
    for key, state in optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(
                value,
                expected_optimizer.state_dict()["state"][key][name],
                rtol=0,
                atol=0,
            )
    before = (tmp_path / "resumed" / "latest.pth").read_bytes()
    training_case(tmp_path / "resumed", resume=True)
    assert (tmp_path / "resumed" / "latest.pth").read_bytes() == before


def test_policy_is_optional_and_rejects_unknown_or_invalid_limits():
    policy = ExecutionPolicy()
    assert policy.tracking.wandb == "disabled" and not policy.tracking.tensorboard
    with pytest.raises(ValueError):
        ExecutionPolicy.model_validate({"budget": {"max_parallel_jobs": 0}})
    with pytest.raises(ValueError):
        ExecutionPolicy.model_validate({"budget": {"max_runz": 2}})


def test_gpu_lease_is_exclusive_and_released(tmp_path, monkeypatch):
    import tsflab.experiments.infra.resources as resource_module

    monkeypatch.setenv("TSFLAB_RESOURCE_DIR", str(tmp_path))
    monkeypatch.setattr(
        resource_module,
        "gpu_inventory",
        lambda: [{"index": "3", "uuid": "GPU-test", "free_mb": "8000"}],
    )
    resources = ExecutionPolicy.model_validate(
        {"resources": {"gpus": ["3"], "wait_timeout_minutes": 0.001}}
    ).resources
    with resource_module.lease_gpus(resources) as devices:
        assert devices == ["GPU-test"]
        with pytest.raises(TimeoutError):
            with resource_module.lease_gpus(resources):
                pass
    with resource_module.lease_gpus(resources) as devices:
        assert devices == ["GPU-test"]


def test_parallel_round_budget_and_resume_do_not_double_count(tmp_path, monkeypatch):
    from tsflab.research.rounds import (
        ResearchRoundBusy,
        claim_run,
        create_round,
        finish_run,
        load_round,
    )

    monkeypatch.setenv("TSFLAB_WORK_DIR", str(tmp_path))
    state = create_round(
        task="test", goal="bounded runs", max_runs=1, budget={"max_parallel_jobs": 1}
    )
    number = claim_run(state["id"], {"run_id": "a", "gpus": 1}, active=True)
    with pytest.raises(ResearchRoundBusy):
        claim_run(state["id"], {"run_id": "b"}, active=True)
    finish_run(state["id"], number, status="failed")
    assert claim_run(state["id"], {"run_id": "a"}, active=True) == number
    assert load_round(state["id"])["runs_used"] == 1
    finish_run(state["id"], number, status="passed")
    assert load_round(state["id"])["gpu_hours_used"] > 0


def test_csv_writes_are_concurrent_and_idempotent(tmp_path):
    import csv

    from tsflab.experiments.utils.results import write_csv_summary

    path = tmp_path / "performance.csv"
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(
            pool.map(
                lambda i: write_csv_summary(
                    str(path), {"run_id": str(i % 4), "mse": i}
                ),
                range(24),
            )
        )
    rows = list(csv.DictReader(path.open()))
    assert len(rows) == 4 and {r["run_id"] for r in rows} == {"0", "1", "2", "3"}


def test_comparison_separates_protocols_and_exposes_missing_or_duplicate_seeds():
    def row(model, seed, protocol="p", variant="v"):
        return {
            "model": model,
            "seed": seed,
            "protocol_sha256": protocol,
            "model_variant": variant,
            "mse": 1,
            "mae": 1,
        }

    result = compare_rows(
        [
            row("a", 0),
            row("a", 1),
            row("b", 0),
            row("c", 0, "different"),
            {"model": "legacy"},
        ]
    )
    assert len(result["cohorts"]) == 2 and result["unverified_runs"] == 1
    cohort = next(c for c in result["cohorts"] if c["protocol"] == "p")
    assert cohort["leaderboard"][0]["rankable"]
    assert cohort["leaderboard"][1]["missing_seeds"] == ["1"]
    duplicated = compare_rows([row("a", 0), row("a", 0)])
    assert not duplicated["cohorts"][0]["leaderboard"][0]["rankable"]


def test_report_propagates_aggregation_failure_without_reading_stale_csv(
    tmp_path, monkeypatch
):
    from tsflab.cli.commands import report

    monkeypatch.setattr(
        "sys.argv", ["report", "--dataset", "fixture", "--work-dir", str(tmp_path)]
    )
    monkeypatch.setattr(
        report,
        "_run_tool",
        lambda *args: SimpleNamespace(
            returncode=7, stderr="aggregation failed", stdout=""
        ),
    )
    assert report.main() == 7
    assert not (tmp_path / "fixture" / "report.md").exists()


def test_run_records_failure_before_data_loading_and_rejects_changed_data(
    tmp_path, monkeypatch
):
    import importlib

    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra.runs import prepare_run, read_run, verify_resume

    module = importlib.import_module("tsflab.experiments.runner.run_one")
    loaded = load_config("configs/runs/smoke_crib.toml")[0]
    config = loaded.config
    config.experiment.work_dir = str(tmp_path)
    directory = prepare_run(config, loaded.raw)
    monkeypatch.setenv("TSFLAB_RUN_DIR", str(directory))
    monkeypatch.setattr(
        module,
        "_build_loaders",
        lambda *args: (_ for _ in ()).throw(RuntimeError("loader exploded")),
    )
    with pytest.raises(RuntimeError, match="loader exploded"):
        module.run_one(config, loaded.raw)
    record = read_run(directory)
    assert record["status"] == "failed" and record["stage"] == "data"
    assert record["attempts"][0]["status"] == "failed"
    config.task.pred_len += 1
    with pytest.raises(ValueError, match="configuration changed"):
        verify_resume(directory, config)


def test_default_tracking_has_no_optional_imports(tmp_path, monkeypatch):
    import sys

    from tsflab.experiments.infra.tracking import Tracker

    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.setitem(sys.modules, "tensorboard", None)
    tracker = Tracker(tmp_path, "run", {}, ExecutionPolicy().tracking)
    tracker.log({"train/loss": 0.5}, 1)
    tracker.close()
    assert json.loads((tmp_path / "events.jsonl").read_text())["metrics"] == {
        "train/loss": 0.5
    }


def test_real_optional_tracking_writes_tensorboard_and_wandb_offline(tmp_path):
    pytest.importorskip("tensorboard")
    pytest.importorskip("wandb")
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    from tsflab.experiments.infra.tracking import Tracker

    tracker = Tracker(
        tmp_path,
        "offline-fixture",
        {"seed": 17},
        ExecutionPolicy.model_validate(
            {"tracking": {"tensorboard": True, "wandb": "offline"}}
        ).tracking,
    )
    tracker.log({"train/loss": 0.25, "validation/loss": 0.5}, 1)
    tracker.close()
    accumulator = EventAccumulator(str(tmp_path / "tensorboard")).Reload()
    assert accumulator.Scalars("train/loss")[0].value == 0.25
    assert list((tmp_path / "wandb").rglob("*.wandb"))


def test_tracker_failure_preserves_local_events_and_closes_backend(tmp_path):
    from tsflab.experiments.infra.tracking import Tracker

    class BrokenWriter:
        closed = False

        def add_scalar(self, *args):
            raise RuntimeError("backend disconnected")

        def close(self):
            self.closed = True

    backend = BrokenWriter()
    tracker = Tracker(tmp_path, "failure", {}, ExecutionPolicy().tracking)
    tracker.started = True
    tracker.writer = backend
    with pytest.warns(UserWarning, match="local events"):
        tracker.log({"train/loss": 0.5}, 1)
    tracker.log({"train/loss": 0.4}, 2)
    tracker.close()
    assert backend.closed
    assert len((tmp_path / "events.jsonl").read_text().splitlines()) == 2


def test_planned_cells_with_no_results_remain_in_comparison():
    plan = [
        {
            "model": model,
            "model_variant": model,
            "protocol_sha256": "same",
            "seed": seed,
        }
        for model in ("a", "b")
        for seed in (0, 1)
    ]
    result = compare_rows([{**plan[0], "mse": 0.5, "mae": 0.25}], planned=plan)
    rows = result["cohorts"][0]["leaderboard"]
    assert len(rows) == 2
    assert not any(row["rankable"] for row in rows)
    assert next(row for row in rows if row["model"] == "b")["missing_seeds"] == [
        "0",
        "1",
    ]


def test_batch_recovery_matches_uninterrupted(tmp_path, monkeypatch):
    import tsflab.experiments.infra.checkpoint as checkpoints

    original_train = train
    monkeypatch.setitem(
        training_case.__globals__,
        "train",
        lambda *args, **kwargs: original_train(
            *args, checkpoint_every_batches=1, **kwargs
        ),
    )
    expected, _, _ = training_case(tmp_path / "full")
    save = checkpoints.save_checkpoint

    def interrupt(*args, **kwargs):
        save(*args, **kwargs)
        if kwargs.get("progress", {}).get("next_batch") == 2:
            raise RuntimeError("batch interruption")

    monkeypatch.setattr(checkpoints, "save_checkpoint", interrupt)
    with pytest.raises(RuntimeError, match="batch interruption"):
        training_case(tmp_path / "partial")
    monkeypatch.setattr(checkpoints, "save_checkpoint", save)
    actual, _, _ = training_case(tmp_path / "partial", resume=True)
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(actual.state_dict()[key], value, rtol=0, atol=0)


def test_external_usage_reservation_is_atomic_and_idempotent(tmp_path):
    from tsflab.experiments.infra.accounting import account

    budget = ExecutionPolicy.model_validate(
        {"budget": {"max_tokens": 100, "max_cost_usd": 1}}
    ).budget

    def reserve(i):
        try:
            account(tmp_path, "reserve", str(i), tokens=60, cost_usd=0.6, budget=budget)
            return True
        except ValueError:
            return False

    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(reserve, [0, 1]))
    assert sum(results) == 1
    key = str(results.index(True))
    account(tmp_path, "settle", key, tokens=40, cost_usd=0.4)
    account(tmp_path, "settle", key, tokens=40, cost_usd=0.4)
    assert account(tmp_path, "status")["totals"]["tokens"] == 40
    with pytest.raises(ValueError, match="immutable"):
        account(tmp_path, "settle", key, tokens=20)


def test_storage_preview_protects_live_and_best_files(tmp_path):
    from tsflab.experiments.infra.retention import cleanup
    from tsflab.experiments.infra.storage import file_lock

    (tmp_path / "manifest.json").write_text("{}")
    folder = tmp_path / "checkpoints"
    folder.mkdir()
    for name in ["epoch_1.pth", "epoch_2.pth", "best_checkpoint.pth"]:
        (folder / name).write_text("weights")
    policy = ExecutionPolicy.model_validate({"storage": {"keep_epoch_checkpoints": 1}})
    plan = cleanup(tmp_path, policy)
    assert plan["files"] == [str(folder / "epoch_1.pth")]
    assert (folder / "epoch_1.pth").exists()
    with file_lock(tmp_path / ".run.lock"):
        with pytest.raises(BlockingIOError):
            cleanup(tmp_path, policy, apply=True)
    cleanup(tmp_path, policy, apply=True)
    assert (folder / "best_checkpoint.pth").exists()
    assert not (folder / "epoch_1.pth").exists()


def test_shared_gpu_limits_and_exclusive_compatibility(tmp_path, monkeypatch):
    import tsflab.experiments.infra.resources as resources

    monkeypatch.setenv("TSFLAB_RESOURCE_DIR", str(tmp_path))
    monkeypatch.setattr(
        resources,
        "gpu_inventory",
        lambda: [{"index": "0", "uuid": "test", "free_mb": "8000", "total_mb": "8000"}],
    )
    policy = ExecutionPolicy.model_validate(
        {
            "resources": {
                "gpus": ["0"],
                "sharing": True,
                "memory_per_run_mb": 3000,
                "wait_timeout_minutes": 0.001,
            }
        }
    )
    with resources.lease_gpus(policy.resources):
        with resources.lease_gpus(policy.resources):
            with pytest.raises(TimeoutError):
                with resources.lease_gpus(policy.resources):
                    pass
        with pytest.raises(TimeoutError):
            with resources.lease_gpus(
                policy.resources.model_copy(update={"sharing": False})
            ):
                pass
    with resources.lease_gpus(policy.resources.model_copy(update={"sharing": False})):
        pass


def test_queue_priority_and_duplicate_submission(tmp_path, monkeypatch):
    from tsflab.experiments.infra.queue import enqueue, jobs

    monkeypatch.setattr("tsflab.experiments.infra.execution.status", lambda p: {})
    low = enqueue(tmp_path, tmp_path / "low", priority=1)
    high = enqueue(tmp_path, tmp_path / "high", priority=10)
    assert enqueue(tmp_path, tmp_path / "low")["id"] == low["id"]
    assert [j["id"] for j in jobs(tmp_path)] == [high["id"], low["id"]]


def test_slurm_uses_argument_arrays_and_retains_job_identity(tmp_path, monkeypatch):
    from tsflab.experiments.infra.slurm import slurm

    (tmp_path / "sweep.json").write_text("{}")
    calls = []

    def command(args, **kwargs):
        calls.append(args)
        return SimpleNamespace(
            stdout="123;cluster\n" if args[0] == "sbatch" else "123|COMPLETED|0:0\n"
        )

    monkeypatch.setattr("tsflab.experiments.infra.slurm.subprocess.run", command)
    assert slurm(tmp_path, "submit", partition="a;echo bad")["job_id"] == "123"
    assert "a;echo bad" in calls[0]
    with pytest.raises(ValueError, match="already"):
        slurm(tmp_path, "submit")
    assert slurm(tmp_path, "status")["records"][0]["state"] == "COMPLETED"
    slurm(tmp_path, "cancel")
    assert calls[-1] == ["scancel", "--clusters", "cluster", "123"]


def test_cli_envelope_includes_parser_failures():
    import subprocess
    import sys

    result = subprocess.run(
        [sys.executable, "-m", "tsflab.cli.main", "--format", "json", "run", "--invalid"],
        text=True,
        capture_output=True,
    )
    payload = json.loads(result.stdout)
    assert not payload["ok"] and payload["exit_code"] == 2
    assert payload["error"]["message"]


def test_runtime_state_contract_rejects_partial_hooks():
    from tsflab.experiments.infra.checkpoint import restore_runtime_state, runtime_state

    model = Forecaster()
    model.runtime_state_dict = lambda: {"cursor": 4}
    with pytest.raises(ValueError, match="both"):
        runtime_state(model)
    model.load_runtime_state_dict = lambda state: setattr(
        model, "cursor", state["cursor"]
    )
    restore_runtime_state(model, runtime_state(model))
    assert model.cursor == 4


def test_evaluation_resumes_predictions_and_rng(tmp_path, monkeypatch):
    import tsflab.experiments.infra.stages as stages
    from tsflab.experiments.runner.evaluator import evaluate

    def run(checkpoint):
        torch.manual_seed(43)
        model = Forecaster()
        loader = DataLoader(
            TensorDataset(
                torch.ones(8, 4, 1),
                torch.ones(8, 2, 1),
                torch.zeros(8, 4, 1),
                torch.zeros(8, 2, 1),
            ),
            batch_size=2,
        )
        return evaluate(
            model,
            loader,
            torch.device("cpu"),
            0,
            2,
            "M",
            checkpoint=checkpoint,
            checkpoint_every_batches=1,
        )[0]

    expected = run(None)
    save = stages.atomic_state

    def interrupted(path, state):
        save(path, state)
        if state["next_batch"] == 2:
            raise RuntimeError("evaluation interrupted")

    monkeypatch.setattr(stages, "atomic_state", interrupted)
    with pytest.raises(RuntimeError, match="evaluation interrupted"):
        run(tmp_path / "evaluation.pth")
    monkeypatch.setattr(stages, "atomic_state", save)
    actual = run(tmp_path / "evaluation.pth")
    import numpy as np

    np.testing.assert_allclose(
        list(actual.values()), list(expected.values()), rtol=0, atol=0, equal_nan=True
    )


# ---------------------------------------------------------------------------
# Infrastructure module boundaries
# ---------------------------------------------------------------------------


def test_leaf_services_work_without_training_or_orchestration_imports(tmp_path):
    script = """
import importlib.abc
import sys
from pathlib import Path
class Boundary(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = ('torch', 'wandb', 'tensorboard', 'tsflab.experiments.runner',
                     'tsflab.experiments.infra.execution', 'tsflab.experiments.infra.environment',
                     'tsflab.experiments.infra.runs')
        if any(fullname == name or fullname.startswith(name + '.') for name in forbidden):
            raise AssertionError('unwanted dependency: ' + fullname)
sys.meta_path.insert(0, Boundary())
from tsflab.experiments.infra.api import (Tracker, UsageLedger, Budget, Storage,
    Resources, lease_gpus, storage_status, describe_modules, enqueue, run_job)
root = Path(sys.argv[1])
with Tracker(root / 'metrics', 'standalone') as tracker:
    tracker.log({'loss': .25}, 1)
ledger = UsageLedger(root / 'ledger', Budget(max_tokens=10))
ledger.reserve('call', tokens=10)
ledger.settle('call', tokens=4)
assert ledger.status()['totals']['tokens'] == 4
assert storage_status(root / 'metrics', Storage(max_run_gb=1))['ok']
with lease_gpus(Resources(gpus=['0']), directory=root / 'leases',
    inventory=lambda: [{'index':'0', 'uuid':'test', 'free_mb':100}]) as assigned:
    assert assigned == ['test']
item = enqueue(root / 'queue', root / 'input', validate=lambda path: None)
run_job(root / 'queue' / item['id'], executor=lambda path, cancelled: {'ok': not cancelled()})
assert describe_modules()['modules']
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_tracking_has_explicit_idempotent_lifecycle(tmp_path):
    path = tmp_path / "new" / "events"
    with Tracker(path, "test") as tracker:
        tracker.log({"loss": 0.5}, 2)
    tracker.close()
    assert json.loads((path / "events.jsonl").read_text())["metrics"]["loss"] == 0.5
    import pytest

    with pytest.raises(RuntimeError, match="closed"):
        tracker.log({"loss": 0}, 3)


def test_injected_queue_execution_has_independent_cancellation(tmp_path):
    from tsflab.experiments.infra.queue import cancel_job, enqueue, jobs, run_job

    root = tmp_path / "queue"
    items = [
        enqueue(root, tmp_path / name, validate=lambda path: None)
        for name in ("a", "b")
    ]
    before = dict(os.environ)
    entered = threading.Barrier(3)
    released = threading.Event()
    seen = {}

    def executor(directory, *, cancelled):
        entered.wait(timeout=10)
        assert released.wait(timeout=10)
        seen[Path(directory).name] = cancelled()
        return {"ok": not cancelled()}

    with ThreadPoolExecutor(2) as pool:
        futures = [
            pool.submit(run_job, root / item["id"], executor=executor) for item in items
        ]
        entered.wait(timeout=10)
        cancel_job(root / items[0]["id"])
        released.set()
        for future in futures:
            future.result(timeout=10)
    assert seen == {"a": True, "b": False}
    assert {Path(j["directory"]).name: j["status"] for j in jobs(root)} == {
        "a": "cancelled",
        "b": "succeeded",
    }
    assert dict(os.environ) == before


def test_module_discovery_matches_real_exports_and_schema():
    import importlib

    for module in describe_modules()["modules"]:
        loaded = importlib.import_module(module["import"])
        for name in module["exports"]:
            assert callable(getattr(loaded, name)), (module["name"], name)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tsflab.cli.main",
            "agent",
            "interface",
            "schema",
            "--module",
            "storage",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert set(json.loads(result.stdout)["properties"]) == {
        "max_run_gb",
        "keep_epoch_checkpoints",
    }


def test_cancellation_composition_is_read_only_until_requested(tmp_path):
    token = FileCancellation(tmp_path / "nested" / "cancel")
    combined = any_cancelled(lambda: False, token)
    assert not combined()
    assert not token.path.parent.exists()
    token.request()
    assert combined()


def test_storage_scope_accepts_local_options_and_rejects_missing_path(tmp_path):
    import pytest

    assert storage_status(tmp_path, Storage())["ok"]
    with pytest.raises(ValueError, match="existing directory"):
        storage_status(tmp_path / "missing")


def test_ledger_facade_keeps_limits_across_instances(tmp_path):
    import pytest

    first = UsageLedger(tmp_path, Budget(max_tokens=10))
    first.reserve("one", tokens=8)
    with pytest.raises(ValueError, match="exhausted"):
        UsageLedger(tmp_path).reserve("two", tokens=3)
    first.settle("one", tokens=1)
    UsageLedger(tmp_path).reserve("two", tokens=3)


def test_agent_task_preparation_is_optional_and_nonpersistent(tmp_path, monkeypatch):
    from tsflab.experiments.infra.api import prepare_task

    target = tmp_path / "workspace"
    monkeypatch.setenv("TSFLAB_WORK_DIR", str(target))
    prepared = prepare_task("autoresearch", {"question": "test", "max_runs": "1"})
    assert prepared["round"] is None and prepared["prompt_path"] is None
    assert prepared["dispatch"] == "not-performed"
    assert prepared["task"]["budget"]["max_runs"] == 1
    assert not target.exists()


def test_agent_api_and_cli_share_persistent_budget_service(
    tmp_path, monkeypatch, capsys
):
    import pytest

    from tsflab.cli.main import main
    from tsflab.experiments.infra.api import load_round, prepare_task
    from tsflab.research.rounds import ResearchRoundError, claim_run

    monkeypatch.setenv("TSFLAB_WORK_DIR", str(tmp_path))
    prepared = prepare_task(
        "autoresearch", {"question": "test", "max_runs": "1"}, persist=True
    )
    round_id = prepared["round"]["id"]
    assert Path(prepared["prompt_path"]).exists()
    claim_run(round_id, {"run_id": "first"})
    with pytest.raises(ResearchRoundError):
        claim_run(round_id, {"run_id": "second"})
    assert (
        main(
            [
                "agent",
                "task",
                "start",
                "autoresearch",
                "--set",
                "question=test",
                "--set",
                "max_runs=1",
                "--json",
            ]
        )
        == 0
    )
    cli = json.loads(capsys.readouterr().out)
    assert cli["task"] == prepared["task"]
    assert load_round(cli["round"]["id"])["budget"] == prepared["round"]["budget"]
    assert cli["dispatch"] == "not-performed"


# ---------------------------------------------------------------------------
# Budgets, iterations, and executor parity
# ---------------------------------------------------------------------------


def test_budget_has_one_validated_source_before_persistence(tmp_path, monkeypatch):
    from tsflab.research.rounds import (
        ResearchRoundError,
        claim_run,
        create_round,
        load_round,
    )

    monkeypatch.setenv("TSFLAB_WORK_DIR", str(tmp_path))
    for kwargs in (
        {"max_runs": 2, "budget": {"max_runs": 3}},
        {"budget": {"max_runs": -1}},
        {"budget": {"max_runs": True}},
        {"budget": {"max_gpu_hours": float("nan")}},
    ):
        with pytest.raises(ResearchRoundError):
            create_round(task="test", goal="test", **kwargs)
    assert not list(tmp_path.iterdir())
    state = create_round(task="test", goal="test", budget={"max_runs": 1})
    assert state["max_runs"] == state["budget"]["max_runs"] == 1
    claim_run(state["id"], {"run_id": "one"})
    with pytest.raises(ResearchRoundError, match="exhausted"):
        claim_run(state["id"], {"run_id": "two"})
    assert load_round(state["id"])["runs_used"] == 1


def test_matrix_preparation_does_not_define_research_iterations(tmp_path, monkeypatch):
    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra.execution import prepare_sweep
    from tsflab.research.rounds import (
        ResearchRoundError,
        claim_iteration,
        create_round,
        load_round,
    )

    monkeypatch.setenv("TSFLAB_WORK_DIR", str(tmp_path))
    state = create_round(task="test", goal="test", budget={"max_iterations": 1})
    loaded = load_config("configs/runs/smoke_crib.toml")
    loaded[0].config.experiment.work_dir = str(tmp_path)
    prepare_sweep(loaded, ExecutionPolicy(), state["id"])
    assert load_round(state["id"])["iterations_used"] == 0
    claim_iteration(state["id"], operation="hypothesis-a")
    prepare_sweep(loaded, ExecutionPolicy(), state["id"])
    claim_iteration(state["id"], operation="hypothesis-a")
    assert load_round(state["id"])["iterations_used"] == 1
    with pytest.raises(ResearchRoundError, match="exhausted"):
        claim_iteration(state["id"], operation="hypothesis-b")


def test_preflight_returns_resolved_copy_and_preserves_input(tmp_path, monkeypatch):
    from tsflab.experiments.config.loader import load_config
    from tsflab.experiments.infra import execution

    config = load_config("configs/runs/smoke_crib.toml")[0].config
    config.experiment.runtime.device = "cuda"
    config.experiment.runtime.device_ids = [0]
    policy = ExecutionPolicy()
    original = policy.model_dump()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(
        "tsflab.experiments.infra.environment.gpu_inventory", lambda: [{"uuid": "gpu-test"}]
    )
    monkeypatch.setattr(execution, "audit_environment", lambda *a, **k: {"checks": []})
    report = execution.preflight([config], policy)
    assert report["ok"]
    assert report["resolved_policy"]["resources"]["gpus"] == ["gpu-test"]
    assert policy.model_dump() == original
    report["resolved_policy"]["resources"]["gpus"].clear()
    assert policy.model_dump() == original


def test_same_executor_contract_in_process_and_detached(tmp_path, monkeypatch):
    from tsflab.experiments.infra.queue import enqueue, jobs, run_job, work

    module = tmp_path / "local_test_executor.py"
    module.write_text(
        'def execute(directory, *, cancelled):\n    return {"ok": not cancelled(), "value": 17}\n'
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    import os

    monkeypatch.setenv(
        "PYTHONPATH", str(tmp_path) + os.pathsep + os.environ.get("PYTHONPATH", "")
    )
    root = tmp_path / "queue"
    first = enqueue(
        root,
        tmp_path / "one",
        validate=lambda p: None,
        executor="local_test_executor:execute",
    )
    local = run_job(root / first["id"])
    second = enqueue(
        root,
        tmp_path / "two",
        validate=lambda p: None,
        executor="local_test_executor:execute",
    )
    work(root, once=True)
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        remote = next(job for job in jobs(root) if job["id"] == second["id"])
        if remote["status"] in {"succeeded", "failed"}:
            break
        time.sleep(0.05)
    assert remote["status"] == local["status"] == "succeeded", remote
    assert remote["result"] == local["result"]


def test_executor_contract_errors_are_structured(tmp_path):
    from tsflab.experiments.infra.queue import enqueue, run_job

    item = enqueue(tmp_path, tmp_path / "input", validate=lambda p: None)
    state = run_job(tmp_path / item["id"], executor=lambda *a, **k: {"ok": "yes"})
    assert state["status"] == "failed"
    assert state["error"]["type"] == "ContractError"


def test_cli_envelope_calls_route_directly_and_shares_result_shape(monkeypatch, capsys):
    from tsflab.cli.main import main
    from tsflab.experiments.infra.results import invoke

    def no_subprocess(*a, **k):
        raise AssertionError("unexpected CLI subprocess")

    monkeypatch.setattr(subprocess, "run", no_subprocess)
    assert main(["--format", "json", "agent", "interface", "schema", "--module", "storage"]) == 0
    actual = json.loads(capsys.readouterr().out)
    expected = invoke(lambda: actual["data"]).to_dict()
    assert {key: actual[key] for key in expected} == expected
    assert main(["--format", "json", "run", "--bad-flag"]) == 2
    assert json.loads(capsys.readouterr().out)["error"]


def test_reconstruction_stage_resume_matches_uninterrupted(tmp_path, monkeypatch):
    """Model-owned pretraining resumes after a complete optimizer step (fake autoencoder)."""
    from functools import partial

    import tsflab.experiments.infra.stages as stages

    def run(path):
        torch.manual_seed(77)
        autoencoder = torch.nn.Sequential(torch.nn.Linear(1, 3), torch.nn.Dropout(0.2), torch.nn.Linear(3, 1))
        loader = DataLoader(
            TensorDataset(torch.arange(32).reshape(8, 4, 1).float() / 32),
            batch_size=2,
            shuffle=True,
        )
        optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.01)
        runner = partial(stages.reconstruction_stage, checkpoint=path, every_batches=1)
        runner(autoencoder, loader, torch.device("cpu"), optimizer, torch.nn.MSELoss(), 2)
        return autoencoder

    expected = run(tmp_path / "full.pth")
    save = stages.atomic_state

    def interrupted(path, state):
        save(path, state)
        if state["next_batch"] == 2:
            raise RuntimeError("interrupted stage")

    monkeypatch.setattr(stages, "atomic_state", interrupted)
    with pytest.raises(RuntimeError, match="interrupted stage"):
        run(tmp_path / "resumed.pth")
    monkeypatch.setattr(stages, "atomic_state", save)
    actual = run(tmp_path / "resumed.pth")
    for key, value in expected.state_dict().items():
        torch.testing.assert_close(value, actual.state_dict()[key], rtol=0, atol=0)


# ---------------------------------------------------------------------------
# Model I/O helpers
# ---------------------------------------------------------------------------


def test_model_io_preserves_probabilistic_axis():
    from tsflab.experiments.runner.model_io import slice_prediction_target

    output, target = slice_prediction_target(torch.randn(2, 12, 4, 9), torch.randn(2, 16, 4), 6, "MS")
    assert output.shape == (2, 6, 1, 9) and target.shape == (2, 6, 1)


def test_model_io_does_not_mask_internal_type_errors():
    from tsflab.experiments.runner.model_io import call_forecaster

    class Broken(torch.nn.Module):
        def forward(self, x, x_mark_enc=None, x_dec=None, x_mark_dec=None):
            raise TypeError("internal failure")

    values = torch.randn(1, 4, 2)
    with pytest.raises(TypeError, match="internal failure"):
        call_forecaster(Broken(), values, None, values, None)


# --- streaming metrics ---------------------------------------------------------------

def _split(arr, sizes):
    out, start = [], 0
    for size in sizes:
        out.append(arr[start:start + size])
        start += size
    return out


@pytest.mark.parametrize("output_type", ["point", "quantile", "distribution"])
def test_streaming_metrics_match_concatenated(output_type):
    import numpy as np

    from tsflab.experiments.evaluation.streaming import MetricAccumulator
    from tsflab.experiments.runner.evaluator import _CANONICAL_LEVELS, _compute_metrics

    rng = np.random.default_rng(0)
    b, horizon, c = 37, 5, 4
    true = (rng.normal(size=(b, horizon, c)) * 3 + 10).astype(np.float32)
    levels = list(_CANONICAL_LEVELS)
    if output_type == "point":
        pred = (true + rng.normal(size=true.shape)).astype(np.float32)
    elif output_type == "quantile":
        base = true[..., None] + rng.normal(size=(*true.shape, 1))
        pred = np.sort(base + rng.normal(size=(*true.shape, len(levels))), axis=-1).astype(np.float32)
    else:
        loc = true + rng.normal(size=true.shape)
        pred = np.stack([loc, np.abs(rng.normal(size=true.shape)) + 0.5], axis=-1).astype(np.float32)
    expected = _compute_metrics(pred, true, output_type, "gaussian", levels)
    acc = MetricAccumulator(output_type, "gaussian", levels)
    for p, t in zip(_split(pred, [1, 16, 3, 17]), _split(true, [1, 16, 3, 17])):
        acc.update(p, t)
    got = acc.result()
    assert set(got) == set(expected)
    for key, value in expected.items():
        assert got[key] == pytest.approx(value, rel=1e-5, abs=1e-6), key


def test_streaming_metrics_zero_variance_and_empty():
    import numpy as np

    from tsflab.experiments.evaluation.streaming import MetricAccumulator
    from tsflab.experiments.evaluation.metrics import collect_metrics

    true = np.ones((4, 3, 2), dtype=np.float32)
    pred = np.full_like(true, 2.0)
    acc = MetricAccumulator()
    acc.update(pred[:2], true[:2])
    acc.update(pred[2:], true[2:])
    got, expected = acc.result(), collect_metrics(pred, true)
    assert got["corr"] == expected["corr"] == 0.0
    assert np.isnan(got["mase"]) and np.isnan(expected["mase"])
    with pytest.raises(ValueError):
        MetricAccumulator().result()


def test_profile_runs_every_forward_without_autograd(tmp_path):
    """An iterative sampler under autograd keeps all steps alive (NsDiff OOM in the smoke)."""
    from tsflab.experiments.evaluation.profile import profile_model

    class Recorder(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 3)
            self.grad_modes = []

        def forward(self, x, x_mark=None, x_dec=None, x_mark_dec=None):
            self.grad_modes.append(torch.is_grad_enabled())
            return self.proj(x)[:, -2:]

    loader = DataLoader(TensorDataset(torch.randn(4, 5, 3), torch.randn(4, 4, 3),
                                      torch.zeros(4, 5, 4), torch.zeros(4, 4, 4)), batch_size=2)
    model = Recorder()
    profile_model(model, loader, torch.device("cpu"), label_len=2, pred_len=2,
                  save_path=str(tmp_path / "profile.txt"))
    assert model.grad_modes and not any(model.grad_modes)
    assert "Total MACs" in (tmp_path / "profile.txt").read_text()
