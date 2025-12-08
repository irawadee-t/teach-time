"""Base classes for LLM pipelines."""

import asyncio
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any

from ..models import UseCase


@dataclass
class PipelineResult:
    """Base result class for all pipelines."""
    final_response: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"final_response": self.final_response, **self.metadata}


class BasePipeline(ABC):
    """
    Base class for LLM pipelines (chains, best-of-n, etc.).

    Provides common functionality:
    - Provider management (sync/async detection)
    - Debug output
    - Context extraction from messages
    - Model function wrapping for TutorBench compatibility
    """

    _output_key: str = "_outputs"  # Override in subclasses

    def __init__(self, base_provider: Callable[[str, List[Dict]], str], verbose: bool = False):
        self.base_provider = base_provider
        self.verbose = verbose
        self.is_async = asyncio.iscoroutinefunction(base_provider)

    @abstractmethod
    async def run_async(
        self,
        system_prompt: str,
        messages: List[Dict],
        use_case: Optional[UseCase] = None,
    ) -> PipelineResult:
        """Execute the pipeline. Subclasses implement specific logic."""
        pass

    def create_model_fn(self, log_outputs: bool = False) -> Callable:
        """Create TutorBench-compatible model function."""
        output_key = self._output_key

        async def model_fn(system_prompt: str, messages: List[Dict]) -> str:
            result = await self.run_async(system_prompt, messages)
            if log_outputs:
                if not hasattr(model_fn, output_key):
                    setattr(model_fn, output_key, {})
                outputs = getattr(model_fn, output_key)
                outputs[len(outputs)] = result.to_dict()
            return result.final_response

        def get_outputs():
            return getattr(model_fn, output_key, {})

        model_fn.get_outputs = get_outputs
        return model_fn

    def extract_context(self, messages: List[Dict]) -> Dict[str, str]:
        """Extract question and student work from messages."""
        context = {"question": "", "student_work": ""}
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if isinstance(content, str):
                if role == "user":
                    if not context["question"]:
                        context["question"] = content
                    else:
                        context["student_work"] += "\n" + content
        return context

    async def call_provider(self, system_prompt: str, messages: List[Dict]) -> str:
        """Call provider, handling sync/async transparently."""
        if self.is_async:
            return await self.base_provider(system_prompt, messages)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self.base_provider, system_prompt, messages)

    def debug_print(self, header: str, content: str = ""):
        """Print debug info if verbose mode enabled."""
        if self.verbose:
            print(f"\n{'='*60}")
            print(header)
            print(f"{'='*60}")
            if content:
                print(content[:200] + "..." if len(content) > 200 else content)


def save_pipeline_debug_log(
    outputs: Dict,
    samples: List,
    results: List,
    output_path: str,
    data_key: str = "pipeline_data",
) -> None:
    """
    Generic debug log saver for any pipeline type.

    Args:
        outputs: Dict from model_fn.get_outputs()
        samples: List of Sample objects
        results: List of EvaluationResult objects
        output_path: Path to save debug log
        data_key: Key name for pipeline data in output
    """
    debug_data = {
        "metadata": {
            "total_samples": len(samples),
            "total_outputs": len(outputs),
        },
        "samples": []
    }

    for idx, (sample, result) in enumerate(zip(samples, results)):
        sample_debug = {
            "sample_id": sample.sample_id,
            "use_case": sample.use_case.value,
            "subject": sample.subject,
            "score": result.weighted_score,
            "pass_rate": result.pass_rate,
            "num_rubrics": len(result.rubric_ratings),
            data_key: outputs.get(idx, {}),
        }
        debug_data["samples"].append(sample_debug)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(debug_data, f, indent=2)
    print(f"Debug log saved to: {output_path}")
