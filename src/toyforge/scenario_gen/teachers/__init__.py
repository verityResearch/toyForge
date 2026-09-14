"""Teacher adapters — modular providers behind the `Teacher` protocol."""

from toyforge.scenario_gen.teachers.base import Teacher, TeacherConfig, build_teacher

__all__ = ["Teacher", "TeacherConfig", "build_teacher"]
