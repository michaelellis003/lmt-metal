"""Bridge between lmxlab experiments and gpu-worker job system.

Converts lmxlab model configs and training configs into gpu-worker
job specs, and parses job results back into experiment log entries.

This module is only useful when the gpu-worker service is available
(ml-agent-kit environment). All gpu_worker imports are lazy.

Example::

    from lmxlab.experiments.gpu_bridge import (
        submit_experiment,
        collect_result,
    )

    job_id = submit_experiment(
        arch='llama',
        train_config={'learning_rate': 1e-3, 'max_steps': 200},
        seed=42,
    )

    # Later...
    entry = collect_result(job_id)
"""

from __future__ import annotations

from typing import Any

from lmxlab.core.config import ModelConfig
from lmxlab.experiments.tracking import ExperimentLog, LogEntry
from lmxlab.models.deepseek import deepseek_tiny
from lmxlab.models.gemma import gemma_tiny
from lmxlab.models.gpt import gpt_tiny
from lmxlab.models.llama import llama_tiny
from lmxlab.models.mixtral import mixtral_tiny
from lmxlab.models.qwen import qwen_tiny

ARCH_FACTORIES: dict[str, Any] = {
    "gpt": gpt_tiny,
    "llama": llama_tiny,
    "gemma": gemma_tiny,
    "qwen": qwen_tiny,
    "mixtral": mixtral_tiny,
    "deepseek": deepseek_tiny,
}


def config_to_job_spec(
    arch: str,
    model_config: ModelConfig | None = None,
    train_overrides: dict[str, Any] | None = None,
    seed: int = 42,
    time_budget_s: float = 300.0,
) -> dict[str, Any]:
    """Convert lmxlab config to a gpu-worker job spec dict.

    Args:
        arch: Architecture name (gpt, llama, etc.).
        model_config: Model config. If None, uses tiny preset.
        train_overrides: Override training parameters.
        seed: Random seed.
        time_budget_s: Time budget in seconds.

    Returns:
        Dict suitable for gpu_worker.submit_job().
    """
    if model_config is None:
        factory = ARCH_FACTORIES.get(arch)
        if factory is None:
            raise ValueError(
                f"Unknown arch: {arch}. Available: {', '.join(ARCH_FACTORIES)}"
            )
        model_config = factory()

    block = model_config.block
    train = {
        "learning_rate": 1e-3,
        "batch_size": 4,
        "max_steps": 200,
        "warmup_steps": 5,
        "weight_decay": 0.01,
        "max_grad_norm": 1.0,
    }
    if train_overrides:
        train.update(train_overrides)

    return {
        "model": {
            "type": arch,
            "config": {
                "vocab_size": model_config.vocab_size,
                "d_model": block.d_model,
                "n_heads": block.n_heads,
                "n_kv_heads": block.effective_n_kv_heads,
                "n_layers": model_config.n_layers,
                "d_ff": block.d_ff,
                "attention": block.attention,
                "ffn": block.ffn,
                "norm": block.norm,
                "position": block.position,
                "bias": block.bias,
                "max_seq_len": block.max_seq_len,
            },
        },
        "training": {
            "epochs": 1,
            "batch_size": train["batch_size"],
            "learning_rate": train["learning_rate"],
            "weight_decay": train["weight_decay"],
            "warmup_steps": train["warmup_steps"],
            "max_grad_norm": train["max_grad_norm"],
            "task": "language_modeling",
        },
        "data": {
            "source": "random",
            "seq_length": min(block.max_seq_len, 64),
            "num_samples": 200,
        },
        "device": "auto",
        "metadata": {
            "framework": "mlx",
            "arch": arch,
            "seed": seed,
            "time_budget_s": time_budget_s,
            "lmxlab_config": {
                "attention": block.attention,
                "ffn": block.ffn,
                "norm": block.norm,
                "position": block.position,
            },
        },
    }


def submit_experiment(
    arch: str,
    train_config: dict[str, Any] | None = None,
    seed: int = 42,
    time_budget_s: float = 300.0,
) -> str:
    """Submit a training experiment to the gpu-worker.

    Args:
        arch: Architecture name.
        train_config: Training parameter overrides.
        seed: Random seed.
        time_budget_s: Time budget in seconds.

    Returns:
        Job ID string.

    Raises:
        ImportError: If gpu_worker is not available.
    """
    from gpu_worker import submit_job

    spec = config_to_job_spec(
        arch=arch,
        train_overrides=train_config,
        seed=seed,
        time_budget_s=time_budget_s,
    )
    return submit_job(spec)


def collect_result(
    job_id: str,
    experiment_name: str = "",
) -> LogEntry | None:
    """Collect results from a completed gpu-worker job.

    Converts gpu-worker job results into an ExperimentLog entry.

    Args:
        job_id: Job ID to collect.
        experiment_name: Name for the experiment log entry.

    Returns:
        LogEntry if job is completed, None if still running.

    Raises:
        ImportError: If gpu_worker is not available.
    """
    from gpu_worker import get_results, poll_job

    status = poll_job(job_id)
    if status.get("status") not in ("completed", "failed"):
        return None

    results = get_results(job_id)
    job = results.get("job", {})
    metrics = results.get("metrics", {})

    name = experiment_name or job.get("model", {}).get("type", "unknown")
    final_loss = metrics.get("final_loss", 0.0)

    return LogEntry(
        experiment=name,
        commit="",
        status="crash" if status.get("status") == "failed" else "keep",
        val_loss=float(final_loss),
        train_loss=float(final_loss),
        param_count=0,
        wall_time_s=0.0,
        description=f"gpu-worker job {job_id}",
        config=job,
        metrics=metrics,
        seed=job.get("metadata", {}).get("seed", 42),
    )


def log_result(
    job_id: str,
    log_path: str = "experiments/results.jsonl",
    experiment_name: str = "",
) -> LogEntry | None:
    """Collect and log a gpu-worker job result.

    Convenience function that collects the result and appends
    it to the experiment log.

    Args:
        job_id: Job ID.
        log_path: Path to results.jsonl.
        experiment_name: Name for the log entry.

    Returns:
        LogEntry if logged, None if job not done.
    """
    entry = collect_result(job_id, experiment_name)
    if entry is None:
        return None

    log = ExperimentLog(log_path)
    log.log(entry)
    return entry
