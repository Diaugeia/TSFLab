"""Profiling utilities for model architecture and performance."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import torch

from tsflab.experiments.runner.model_io import call_forecaster, make_decoder_input


def _try_torchinfo_summary(model, input_data):
    """Attempt to render a torchinfo summary for the model.

    Parameters
    ----------
    model : nn.Module
        Model to summarize.
    input_data : tuple | torch.Tensor
        Example inputs for tracing.

    Returns
    -------
    str
        Summary string or error message.
    """
    try:
        from torchinfo import summary

        stats = summary(model, input_data=input_data, verbose=0)
        return str(stats)
    except Exception as exc:
        return f"torchinfo unavailable: {exc}"


def _try_flops(model, input_data):
    """Attempt to compute FLOPs using fvcore.

    Parameters
    ----------
    model : nn.Module
        Model to analyze.
    input_data : tuple | torch.Tensor
        Example inputs for tracing.

    Returns
    -------
    str
        FLOPs summary string or error message.
    """
    try:
        from fvcore.nn import FlopCountAnalysis

        # ``FlopCountAnalysis`` traces lazily, on the first statistic request, so the
        # request must also run without autograd.
        with torch.no_grad():
            flops = FlopCountAnalysis(model, input_data)
            total = flops.total()
        return f"Total MACs: {total / 1e6:.4f} M"
    except Exception as exc:
        return f"FLOPs unavailable: {exc}"


def _latency_benchmark(model, input_data, device: torch.device) -> list[str]:
    """Benchmark model latency on CUDA with warmup and memory stats.

    Parameters
    ----------
    model : nn.Module
        Model to benchmark.
    input_data : tuple | torch.Tensor
        Example inputs for tracing.
    device : torch.device
        Target device.

    Returns
    -------
    list[str]
        Report lines for memory and latency.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return ["Latency benchmark skipped (CUDA not available)"]

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    base_mem = torch.cuda.memory_allocated()

    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
    times = []
    mem_lines: list[str] = []

    with torch.no_grad():
        for i in range(60):
            torch.cuda.synchronize()
            starter.record()
            _ = (
                model(*input_data)
                if isinstance(input_data, tuple)
                else model(input_data)
            )
            ender.record()
            torch.cuda.synchronize()

            res_time = starter.elapsed_time(ender)
            if i == 0:
                peak = torch.cuda.max_memory_allocated()
                reserved_mem = torch.cuda.max_memory_reserved()
                mem_lines = [
                    f"Dynamic VRAM: {(peak - base_mem) / 1024**2:.2f} MB",
                    f"Total Peak VRAM: {peak / 1024**2:.2f} MB",
                    f"Total Reserved VRAM: {reserved_mem / 1024**2:.2f} MB",
                ]
            if i >= 10:
                times.append(res_time)

    if times:
        avg_t = sum(times) / len(times)
        return mem_lines + [
            f"Average Latency: {avg_t:.4f} ms",
            f"Throughput: {1000 / avg_t:.2f} samples/sec",
        ]
    return mem_lines


def profile_model(
    model,
    data_loader: Iterable,
    device: torch.device,
    label_len: int,
    pred_len: int,
    save_path: str,
) -> None:
    """Generate a profiling report for a model and save to disk.

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    data_loader : Iterable
        Data loader providing input batches.
    device : torch.device
        Target device.
    label_len : int
        Decoder label length.
    pred_len : int
        Prediction horizon length.
    save_path : str
        Output path for the profiling report.

    Returns
    -------
    None
    """
    was_training = model.training
    model.eval()
    try:
        model_for_summary = (
            model.module if isinstance(model, torch.nn.DataParallel) else model
        )

        it = iter(data_loader)
        batch_x, batch_y, batch_x_mark, batch_y_mark = next(it)
        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
        if batch_x_mark is not None:
            batch_x_mark = batch_x_mark.float().to(device)
        if batch_y_mark is not None:
            batch_y_mark = batch_y_mark.float().to(device)

        dec_inp = make_decoder_input(batch_y, label_len, pred_len, device)
        input_data = (batch_x, batch_x_mark, dec_inp, batch_y_mark)
        # Inference only: with autograd on, an iterative sampler (diffusion steps,
        # autoregressive decoding) keeps every intermediate of every step alive.
        with torch.no_grad():
            call_forecaster(model, *input_data)

        report = [
            "[Architecture & Parameters]",
            _try_torchinfo_summary(model_for_summary, input_data),
            "\n[FLOPs]",
            _try_flops(model_for_summary, input_data),
            "\n[Performance]",
        ]
        report.extend(_latency_benchmark(model, input_data, device))

        target = Path(save_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("\n".join(report), encoding="utf-8")
    finally:
        model.train(was_training)


def parse_profile_report(report: str) -> dict[str, object]:
    """Parse a profiling report into structured metrics.

    Parameters
    ----------
    report : str
        Full profile report contents.

    Returns
    -------
    dict[str, object]
        Parsed metrics with raw units preserved.
    """
    metrics: dict[str, object] = {}
    prefix_map: dict[str, tuple[str, str]] = {
        "Total params:": ("total_params", "int"),
        "Trainable params:": ("trainable_params", "int"),
        "Non-trainable params:": ("non_trainable_params", "int"),
        "Total mult-adds (Units.MEGABYTES):": ("total_mult_adds_mb", "raw"),
        "Total MACs:": ("total_macs_m", "raw"),
        "Dynamic VRAM:": ("dynamic_vram_mb", "raw"),
        "Total Peak VRAM:": ("peak_vram_mb", "raw"),
        "Total Reserved VRAM:": ("reserved_vram_mb", "raw"),
        "Average Latency:": ("latency_avg_ms", "raw"),
        "Throughput:": ("throughput_samples_sec", "raw"),
    }

    for line in report.splitlines():
        line = line.strip()
        if not line or line.startswith("["):
            continue
        for prefix, (key, parse_type) in prefix_map.items():
            if not line.startswith(prefix):
                continue
            value = line[len(prefix) :].strip()
            if parse_type == "int":
                value = value.replace(",", "")
                try:
                    metrics[key] = int(value)
                except ValueError:
                    metrics[key] = value
            else:
                metrics[key] = value
            break
    return metrics


def parse_profile_report_file(path: str) -> dict[str, object]:
    """Load and parse a profiling report from disk."""
    if not os.path.exists(path):
        return {}
    with open(path, "r") as f:
        report = f.read()
    return parse_profile_report(report)
