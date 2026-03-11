"""Experiment framework: runner, tracking, sweep, analysis, profiling."""

from lmxlab.experiments.analysis import (
    cohens_d,
    compare_experiments,
    compute_statistics,
    confidence_interval,
    simplicity_score,
)
from lmxlab.experiments.gpu_bridge import (
    collect_result,
    config_to_job_spec,
    log_result,
    submit_experiment,
)
from lmxlab.experiments.profiling import (
    benchmark_fn,
    count_parameters_by_module,
    memory_estimate,
    profile_forward,
    profile_generation,
)
from lmxlab.experiments.runner import ExperimentRunner
from lmxlab.experiments.tracking import ExperimentLog, LogEntry

__all__ = [
    "ExperimentLog",
    "ExperimentRunner",
    "LogEntry",
    "benchmark_fn",
    "cohens_d",
    "collect_result",
    "compare_experiments",
    "compute_statistics",
    "config_to_job_spec",
    "confidence_interval",
    "count_parameters_by_module",
    "log_result",
    "memory_estimate",
    "profile_forward",
    "profile_generation",
    "simplicity_score",
    "submit_experiment",
]
