"""
Main LLM-as-a-judge evaluation engine.

Uses DeepSeek V3 to evaluate tutoring conversations across 3 layers:
- Layer 1: 8-dimension tutor response quality
- Layer 2: Question depth analysis
- Layer 3: ICAP student engagement classification

Adapted for new-teach-time architecture.
"""

import json
from typing import Dict, List, Optional
from pathlib import Path

from .prompts import (
    DIMENSION_PROMPTS,
    QUESTION_DEPTH_PROMPT,
    ICAP_ENGAGEMENT_PROMPT,
    OVERALL_SUMMARY_PROMPT,
)
from .metrics import (
    DimensionScore,
    QuestionDepthScore,
    ICAPScore,
    PESComponents,
    calculate_pes,
    get_pes_category,
)


# Default judge model
JUDGE_MODEL_ID = "deepseek-ai/DeepSeek-V3"


class PedagogicalEvaluator:
    """
    LLM-as-a-judge evaluator for tutoring conversations.

    Uses DeepSeek V3 to assess pedagogical quality across multiple dimensions.
    """

    def __init__(
        self,
        llm_client=None,
        judge_model: str = JUDGE_MODEL_ID,
        verbose: bool = False,
    ):
        """
        Initialize the pedagogical evaluator.

        Args:
            llm_client: LLMClient instance from new-teach-time (required)
            judge_model: Model to use for evaluation (default: DeepSeek V3)
            verbose: Whether to print evaluation progress
        """
        if llm_client is None:
            raise ValueError("llm_client is required. Pass an LLMClient instance.")
        
        self.llm_client = llm_client
        self.judge_model = judge_model
        self.verbose = verbose

        if self.verbose:
            print(f"Initialized PedagogicalEvaluator with model: {judge_model}")

    def _format_conversation(self, conversation: List[Dict]) -> str:
        """
        Format conversation for evaluation prompts.

        Args:
            conversation: List of turn dicts with 'role' and 'content'

        Returns:
            Formatted conversation string
        """
        formatted = []
        for i, turn in enumerate(conversation, 1):
            role = turn.get("role", "unknown")
            content = turn.get("content", "")
            formatted.append(f"Turn {i} [{role.upper()}]: {content}")

        return "\n\n".join(formatted)

    def _call_judge(self, prompt: str) -> Dict:
        """
        Call the judge LLM and parse JSON response.

        Args:
            prompt: Evaluation prompt

        Returns:
            Parsed JSON response as dictionary
        """
        messages = [{"role": "user", "content": prompt}]

        response = self.llm_client.chat(
            model=self.judge_model,
            messages=messages,
            temperature=0.3,  # Low temperature for consistent evaluation
            max_tokens=2000,
        )

        # Extract JSON from response (handle markdown code blocks)
        response = response.strip()
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        try:
            return json.loads(response)
        except json.JSONDecodeError as e:
            if self.verbose:
                print(f"Warning: Failed to parse JSON response: {e}")
                print(f"Raw response: {response[:500]}")
            # Return default structure
            return {"score": 3, "justification": "Error parsing response", "evidence": []}

    def evaluate_dimension(
        self, dimension_key: str, conversation: List[Dict]
    ) -> DimensionScore:
        """
        Evaluate a single pedagogical dimension.

        Args:
            dimension_key: Key from DIMENSION_PROMPTS
            conversation: Conversation to evaluate

        Returns:
            DimensionScore object
        """
        if self.verbose:
            print(f"  Evaluating: {DIMENSION_PROMPTS[dimension_key]['name']}")

        dimension_info = DIMENSION_PROMPTS[dimension_key]
        conversation_text = self._format_conversation(conversation)
        prompt = dimension_info["prompt"].format(conversation=conversation_text)

        result = self._call_judge(prompt)

        # Handle student_talk_ratio special case (has estimated_ratio field)
        evidence = result.get("evidence", [])
        if dimension_key == "student_talk_ratio" and "estimated_ratio" in result:
            # Add ratio to evidence
            evidence.append(f"Estimated ratio: {result['estimated_ratio']}")

        return DimensionScore(
            dimension=dimension_key,
            score=result.get("score", 3),
            justification=result.get("justification", ""),
            evidence=evidence,
        )

    def evaluate_question_depth(self, conversation: List[Dict]) -> QuestionDepthScore:
        """
        Evaluate question depth across the conversation.

        Args:
            conversation: Conversation to evaluate

        Returns:
            QuestionDepthScore object
        """
        if self.verbose:
            print("  Evaluating: Question Depth")

        conversation_text = self._format_conversation(conversation)
        prompt = QUESTION_DEPTH_PROMPT.format(conversation=conversation_text)

        result = self._call_judge(prompt)

        return QuestionDepthScore(
            score=result.get("score", 3),
            question_count=result.get("question_count", {}),
            question_examples=result.get("question_examples", {}),
            justification=result.get("justification", ""),
        )

    def evaluate_icap_engagement(self, conversation: List[Dict]) -> ICAPScore:
        """
        Evaluate student engagement using ICAP framework.

        Args:
            conversation: Conversation to evaluate

        Returns:
            ICAPScore object
        """
        if self.verbose:
            print("  Evaluating: ICAP Engagement")

        conversation_text = self._format_conversation(conversation)
        prompt = ICAP_ENGAGEMENT_PROMPT.format(conversation=conversation_text)

        result = self._call_judge(prompt)

        return ICAPScore(
            score=result.get("score", 3),
            engagement_distribution=result.get("engagement_distribution", {}),
            turn_classifications=result.get("turn_classifications", []),
            justification=result.get("justification", ""),
        )

    def evaluate_conversation(self, conversation: List[Dict]) -> PESComponents:
        """
        Perform complete evaluation of a tutoring conversation.

        Args:
            conversation: List of conversation turns

        Returns:
            PESComponents with all scores and analysis
        """
        if self.verbose:
            print(f"\nEvaluating conversation ({len(conversation)} turns)...")
            print("\nLayer 1: 8-Dimension Analysis")

        # Layer 1: Evaluate all 8 dimensions
        dimension_scores = {}
        for dim_key in DIMENSION_PROMPTS.keys():
            dimension_scores[dim_key] = self.evaluate_dimension(dim_key, conversation)

        # Layer 2: Question depth
        if self.verbose:
            print("\nLayer 2: Question Depth Analysis")
        question_depth = self.evaluate_question_depth(conversation)

        # Layer 3: ICAP engagement
        if self.verbose:
            print("\nLayer 3: ICAP Engagement Classification")
        icap_engagement = self.evaluate_icap_engagement(conversation)

        # Generate overall summary
        if self.verbose:
            print("\nGenerating overall summary...")

        scores_summary = self._format_scores_summary(
            dimension_scores, question_depth, icap_engagement
        )
        summary_prompt = OVERALL_SUMMARY_PROMPT.format(scores_summary=scores_summary)
        summary_result = self._call_judge(summary_prompt)

        # Build PESComponents
        components = PESComponents(
            comprehension_probing=dimension_scores["comprehension_probing"],
            background_knowledge=dimension_scores["background_knowledge"],
            guidance_level=dimension_scores["guidance_level"],
            error_feedback=dimension_scores["error_feedback"],
            encouragement=dimension_scores["encouragement"],
            coherence=dimension_scores["coherence"],
            relevance=dimension_scores["relevance"],
            student_talk_ratio=dimension_scores["student_talk_ratio"],
            question_depth=question_depth,
            icap_engagement=icap_engagement,
            overall_quality=summary_result.get("overall_quality", "adequate"),
            strengths=summary_result.get("strengths", []),
            areas_for_improvement=summary_result.get("areas_for_improvement", []),
            recommendations=summary_result.get("recommendations", []),
            summary=summary_result.get("summary", ""),
        )

        if self.verbose:
            pes = calculate_pes(components)
            print(f"\n{'='*60}")
            print(f"Pedagogical Effectiveness Score (PES): {pes}/100")
            print(f"Category: {get_pes_category(pes)}")
            print(f"{'='*60}\n")

        return components

    def _format_scores_summary(
        self,
        dimension_scores: Dict[str, DimensionScore],
        question_depth: QuestionDepthScore,
        icap_engagement: ICAPScore,
    ) -> str:
        """Format all scores for the summary prompt."""
        lines = ["LAYER 1 - 8 Dimensions (Maurya et al., 2025):"]
        for dim_key, score_obj in dimension_scores.items():
            dim_name = DIMENSION_PROMPTS[dim_key]["name"]
            lines.append(f"  - {dim_name}: {score_obj.score}/5")

        lines.append(f"\nLAYER 2 - Question Depth:")
        lines.append(f"  - Score: {question_depth.score}/5")
        lines.append(f"  - Distribution: {question_depth.question_count}")

        lines.append(f"\nLAYER 3 - ICAP Engagement:")
        lines.append(f"  - Score: {icap_engagement.score}/5")
        lines.append(f"  - Distribution: {icap_engagement.engagement_distribution}")

        return "\n".join(lines)

    def evaluate_from_file(self, filepath: Path) -> PESComponents:
        """
        Evaluate a conversation from a JSON file.

        Args:
            filepath: Path to JSON file containing conversation

        Returns:
            PESComponents with all scores and analysis
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        conversation = data.get("conversation", [])
        if not conversation:
            raise ValueError(f"No conversation found in {filepath}")

        return self.evaluate_conversation(conversation)
