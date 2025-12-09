"""
Agent implementations for teaching environment.

Provides StudentAgent and TeacherAgent wrappers for LLM-based interaction.
"""

from .student_agent import StudentAgent, StudentTurn
from .teacher_agent import TeacherAgent, TeacherTurn

__all__ = [
    "StudentAgent",
    "StudentTurn",
    "TeacherAgent", 
    "TeacherTurn",
]
