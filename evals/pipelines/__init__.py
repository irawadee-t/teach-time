"""Pipeline implementations for multi-stage LLM processing."""

from .base import BasePipeline, PipelineResult, save_pipeline_debug_log
from .chain import (
    ChainedPipeline,
    ChainedPipelineResult,
    StageTemplate,
    create_tutoring_chain,
    create_custom_chain,
    compare_single_vs_chain,
    save_chain_debug_log,
    TUTORING_STAGES,
)
from .best_of_n import (
    BestOfNPipeline,
    BestOfNResult,
    create_best_of_n_pipeline,
    save_bon_debug_log,
)

__all__ = [
    # Base
    "BasePipeline",
    "PipelineResult",
    "save_pipeline_debug_log",
    # Chain
    "ChainedPipeline",
    "ChainedPipelineResult",
    "StageTemplate",
    "create_tutoring_chain",
    "create_custom_chain",
    "compare_single_vs_chain",
    "save_chain_debug_log",
    "TUTORING_STAGES",
    # Best of N
    "BestOfNPipeline",
    "BestOfNResult",
    "create_best_of_n_pipeline",
    "save_bon_debug_log",
]
