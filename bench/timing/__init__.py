"""Interleaved latency measurement and the paired statistics over repetitions."""

from .latency import interleaved
from .stats import across_seeds, paired_log_ratio, summary

__all__ = ["across_seeds", "interleaved", "paired_log_ratio", "summary"]
