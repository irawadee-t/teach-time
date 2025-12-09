"""
Agent implementations for teaching environment.

Provides StudentAgent and TeacherAgent wrappers for LLM-based interaction.
Supports both ReAct and Plain teacher modes for ablation studies.
"""

from .student_agent import StudentAgent, StudentTurn
from .teacher_agent import TeacherAgent, TeacherTurn
from .plain_teacher import PlainTeacherAgent, PlainTeacherTurn


def create_teacher(config, llm_client, use_react: bool = True):
    """
    Factory function to create appropriate teacher agent.
    
    Args:
        config: TeacherConfig instance
        llm_client: LLMClient instance
        use_react: If True, use ReAct framework. If False, use plain mode.
        
    Returns:
        TeacherAgent or PlainTeacherAgent
    """
    if use_react:
        return TeacherAgent(llm_client=llm_client, config=config)
    else:
        return PlainTeacherAgent(llm_client=llm_client, config=config)


__all__ = [
    "StudentAgent",
    "StudentTurn",
    "TeacherAgent", 
    "TeacherTurn",
    "PlainTeacherAgent",
    "PlainTeacherTurn",
    "create_teacher",
]
