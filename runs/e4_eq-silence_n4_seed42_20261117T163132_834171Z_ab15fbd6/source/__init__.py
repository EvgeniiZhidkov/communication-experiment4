"""Reproduction of half-duplex protocols, with auditable model execution."""
from .analysis import analyze_file, analyze_run, verify_reference
from .config import PlayerConfig, RunConfig
from .experiment import audit_tools, input_pairs, run_experiment
from .plotting import plot_results
from .protocols import PROTOCOL_MODES, PROTOCOLS, get_protocol

__all__ = ["PlayerConfig", "RunConfig", "run_experiment", "audit_tools", "input_pairs",
           "analyze_file", "analyze_run", "verify_reference", "plot_results",
           "PROTOCOL_MODES", "PROTOCOLS", "get_protocol"]
