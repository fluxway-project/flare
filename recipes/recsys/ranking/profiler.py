"""Shared profiler configuration for ranking Fluxway jobs."""

from __future__ import annotations

import time

PROFILER_LEVELS = frozenset({"summary", "standard", "detailed", "debug"})


def validate_profiler_level(value: str) -> None:
    if value not in PROFILER_LEVELS:
        choices = ", ".join(sorted(PROFILER_LEVELS))
        raise ValueError(f"profiler_level must be one of: {choices}")


def build_ranking_profiler_config(args):
    """Build the shared Fluxway profiler config from recipe CLI settings."""
    if args.profiler_trace_dir is None:
        return None
    if args.profiler_buffer_mib <= 0:
        raise ValueError(
            f"--profiler-buffer-mib must be positive, got {args.profiler_buffer_mib}"
        )
    if args.profiler_clock_sync_rounds <= 0:
        raise ValueError(
            "--profiler-clock-sync-rounds must be positive, got "
            f"{args.profiler_clock_sync_rounds}"
        )
    if args.profiler_max_clock_uncertainty_us <= 0:
        raise ValueError(
            "--profiler-max-clock-uncertainty-us must be positive, got "
            f"{args.profiler_max_clock_uncertainty_us}"
        )
    if args.profiler_skip_initial_steps < 0:
        raise ValueError(
            "--profiler-skip-initial-steps must be non-negative, got "
            f"{args.profiler_skip_initial_steps}"
        )

    from fluxway.profiler import ProfilerConfig, ProfilerDiagnostic, ProfilerLevel

    run_id = args.profiler_run_id or f"flare_jax_{time.time_ns()}"
    config = ProfilerConfig(
        trace_dir=args.profiler_trace_dir,
        run_id=run_id,
        buffer_bytes_per_process=args.profiler_buffer_mib * 1024 * 1024,
        clock_sync_rounds=args.profiler_clock_sync_rounds,
        max_clock_uncertainty_ns=args.profiler_max_clock_uncertainty_us * 1000,
        output_file=args.profiler_output_file,
        diagnostics=frozenset(
            ProfilerDiagnostic(item) for item in args.profiler_diagnostics
        ),
        level=ProfilerLevel(args.profiler_level),
        skip_initial_steps=args.profiler_skip_initial_steps,
        steady_state_only=args.profiler_steady_state_only,
    )
    config.run_dir.mkdir(parents=True, exist_ok=False)
    return config


__all__ = [
    "PROFILER_LEVELS",
    "build_ranking_profiler_config",
    "validate_profiler_level",
]
