"""
Math domain tasks for TeachTime.

Includes algebra, functions, fractions, and probability concepts.
"""

from typing import List
from .base import TaskDomain, TaskSpec, QuizQuestion


class MathDomain(TaskDomain):
    """Primary domain: Fundamental math concepts (algebra, functions, fractions)."""

    def get_domain_name(self) -> str:
        return "mathematics"

    def get_all_tasks(self) -> List[TaskSpec]:
        return [
            self._linear_equations_task(),
            self._quadratic_equations_task(),
            self._function_composition_task(),
            self._fraction_operations_task(),
        ]

    def _linear_equations_task(self) -> TaskSpec:
        """Task on solving linear equations."""
        return TaskSpec(
            task_id="linear_equations",
            topic="Solving Linear Equations",
            description="Learn to solve one-variable linear equations like 3x + 5 = 20.",
            learning_objectives=[
                "Understand the concept of an equation as a balance",
                "Apply inverse operations to isolate the variable",
                "Check solutions by substitution",
            ],
            key_concepts=["equation", "variable", "inverse operations", "isolate", "solution"],
            common_misconceptions=[
                "Subtracting from one side only",
                "Forgetting to apply operations to both sides",
                "Sign errors when moving terms across equals sign",
            ],
            difficulty="easy",
            pre_quiz=[
                QuizQuestion(
                    question="What does it mean to 'solve' an equation?",
                    question_type="multiple_choice",
                    options=[
                        "A) Make the equation more complicated",
                        "B) Find the value of the variable that makes the equation true",
                        "C) Remove all numbers from the equation",
                        "D) Add more variables"
                    ],
                    correct_answer="B",
                    concept="equation_definition"
                ),
                QuizQuestion(
                    question="If x + 5 = 12, what is x?",
                    question_type="multiple_choice",
                    options=["A) 5", "B) 7", "C) 12", "D) 17"],
                    correct_answer="B",
                    concept="basic_solving"
                ),
                QuizQuestion(
                    question="Solve: 2x = 10. Show your work.",
                    question_type="short_answer",
                    correct_answer="x = 5",
                    rubric="Correct answer is x=5. Accept if student shows division by 2.",
                    concept="basic_solving"
                ),
            ],
            post_quiz=[
                QuizQuestion(
                    question="What is the first step to solve 3x + 7 = 22?",
                    question_type="multiple_choice",
                    options=[
                        "A) Divide both sides by 3",
                        "B) Subtract 7 from both sides",
                        "C) Add 7 to both sides",
                        "D) Multiply both sides by 3"
                    ],
                    correct_answer="B",
                    concept="solving_strategy"
                ),
                QuizQuestion(
                    question="If 4x - 6 = 18, what is x?",
                    question_type="multiple_choice",
                    options=["A) 3", "B) 6", "C) 12", "D) 24"],
                    correct_answer="B",
                    concept="multi_step_solving"
                ),
                QuizQuestion(
                    question="Solve: 5x + 3 = 28. Show your work.",
                    question_type="short_answer",
                    correct_answer="x = 5",
                    rubric="Correct answer is x=5. Full credit if work is shown: subtract 3, then divide by 5.",
                    concept="multi_step_solving"
                ),
            ],
            hints=[
                "Remember: what you do to one side, you must do to the other",
                "Work backwards - undo addition/subtraction first, then multiplication/division",
                "Always check your answer by plugging it back into the original equation",
            ],
            examples=[
                "Example: Solve 2x + 3 = 11\nStep 1: Subtract 3 from both sides → 2x = 8\nStep 2: Divide both sides by 2 → x = 4\nCheck: 2(4) + 3 = 8 + 3 = 11 ✓"
            ]
        )

    def _quadratic_equations_task(self) -> TaskSpec:
        """Task on understanding and solving simple quadratic equations."""
        return TaskSpec(
            task_id="quadratic_equations",
            topic="Introduction to Quadratic Equations",
            description="Understand quadratic equations and learn basic solving methods (factoring).",
            learning_objectives=[
                "Recognize the standard form of a quadratic equation",
                "Understand that quadratics can have two solutions",
                "Factor simple quadratics to find solutions",
            ],
            key_concepts=["quadratic", "parabola", "factoring", "zero product property", "roots"],
            common_misconceptions=[
                "Thinking all equations have exactly one solution",
                "Incorrectly factoring or missing negative signs",
                "Not considering both solutions",
            ],
            difficulty="medium",
            pre_quiz=[
                QuizQuestion(
                    question="What is the standard form of a quadratic equation?",
                    question_type="multiple_choice",
                    options=[
                        "A) ax + b = 0",
                        "B) ax² + bx + c = 0",
                        "C) ax³ + bx² + cx + d = 0",
                        "D) a/x + b = 0"
                    ],
                    correct_answer="B",
                    concept="quadratic_form"
                ),
                QuizQuestion(
                    question="How many solutions can a quadratic equation have?",
                    question_type="multiple_choice",
                    options=["A) Always 1", "B) Always 2", "C) 0, 1, or 2", "D) Infinitely many"],
                    correct_answer="C",
                    concept="solution_count"
                ),
                QuizQuestion(
                    question="If (x - 3)(x + 2) = 0, what are the solutions?",
                    question_type="short_answer",
                    correct_answer="x = 3 and x = -2",
                    rubric="Accept variations like '3 and -2' or 'x=3, x=-2'",
                    concept="zero_product_property"
                ),
            ],
            post_quiz=[
                QuizQuestion(
                    question="To solve a quadratic by factoring, what property do we use?",
                    question_type="multiple_choice",
                    options=[
                        "A) Distributive property",
                        "B) Zero product property",
                        "C) Commutative property",
                        "D) Associative property"
                    ],
                    correct_answer="B",
                    concept="solving_method"
                ),
                QuizQuestion(
                    question="Factor: x² + 5x + 6",
                    question_type="multiple_choice",
                    options=[
                        "A) (x + 2)(x + 3)",
                        "B) (x + 1)(x + 6)",
                        "C) (x - 2)(x - 3)",
                        "D) Cannot be factored"
                    ],
                    correct_answer="A",
                    concept="factoring"
                ),
                QuizQuestion(
                    question="Solve: x² - 4 = 0",
                    question_type="short_answer",
                    correct_answer="x = 2 and x = -2",
                    rubric="Accept: '2 and -2', '±2', 'x=2, x=-2'",
                    concept="difference_of_squares"
                ),
            ],
            hints=[
                "Factor by finding two numbers that multiply to c and add to b",
                "Use the zero product property: if AB = 0, then A = 0 or B = 0",
                "Always check both solutions in the original equation",
            ],
            examples=[
                "Example: Solve x² + 5x + 6 = 0\nFactor: (x + 2)(x + 3) = 0\nSolutions: x = -2 or x = -3"
            ]
        )

    def _function_composition_task(self) -> TaskSpec:
        """Task on understanding function composition."""
        return TaskSpec(
            task_id="function_composition",
            topic="Function Composition",
            description="Learn to compose functions and understand notation like f(g(x)).",
            learning_objectives=[
                "Understand what a function is",
                "Compute composed functions f(g(x))",
                "Recognize that order matters in composition",
            ],
            key_concepts=["function", "composition", "input", "output", "notation"],
            common_misconceptions=[
                "Thinking f(g(x)) = g(f(x)) (order doesn't matter)",
                "Confusing composition with multiplication",
                "Not evaluating the inner function first",
            ],
            difficulty="medium",
            pre_quiz=[
                QuizQuestion(
                    question="What does f(x) represent?",
                    question_type="multiple_choice",
                    options=[
                        "A) f times x",
                        "B) A function named f evaluated at input x",
                        "C) f plus x",
                        "D) A fraction f divided by x"
                    ],
                    correct_answer="B",
                    concept="function_notation"
                ),
                QuizQuestion(
                    question="If f(x) = 2x, what is f(3)?",
                    question_type="multiple_choice",
                    options=["A) 5", "B) 6", "C) 8", "D) 23"],
                    correct_answer="B",
                    concept="function_evaluation"
                ),
                QuizQuestion(
                    question="If f(x) = x + 1 and g(x) = 2x, what is f(g(2))?",
                    question_type="short_answer",
                    correct_answer="5",
                    rubric="Correct answer is 5. g(2)=4, then f(4)=5",
                    concept="basic_composition"
                ),
            ],
            post_quiz=[
                QuizQuestion(
                    question="In function composition f(g(x)), which function do you evaluate first?",
                    question_type="multiple_choice",
                    options=[
                        "A) f",
                        "B) g",
                        "C) Either one",
                        "D) Both simultaneously"
                    ],
                    correct_answer="B",
                    concept="composition_order"
                ),
                QuizQuestion(
                    question="If f(x) = x² and g(x) = x + 3, what is f(g(1))?",
                    question_type="multiple_choice",
                    options=["A) 4", "B) 7", "C) 10", "D) 16"],
                    correct_answer="D",
                    concept="composition_evaluation"
                ),
                QuizQuestion(
                    question="If f(x) = 3x and g(x) = x - 2, find f(g(5)). Show your work.",
                    question_type="short_answer",
                    correct_answer="9",
                    rubric="Correct answer is 9. g(5)=3, then f(3)=9",
                    concept="multi_step_composition"
                ),
            ],
            hints=[
                "Always evaluate the inner function first",
                "Think of it like a machine: the output of g becomes the input of f",
                "Write out each step separately to avoid confusion",
            ],
            examples=[
                "Example: If f(x) = x + 1 and g(x) = 2x, find f(g(3))\nStep 1: Evaluate g(3) = 2(3) = 6\nStep 2: Evaluate f(6) = 6 + 1 = 7\nAnswer: 7"
            ]
        )

    def _fraction_operations_task(self) -> TaskSpec:
        """Task on adding and simplifying fractions."""
        return TaskSpec(
            task_id="fraction_operations",
            topic="Adding and Simplifying Fractions",
            description="Learn to add fractions with different denominators and simplify results.",
            learning_objectives=[
                "Find common denominators",
                "Add fractions correctly",
                "Simplify fractions to lowest terms",
            ],
            key_concepts=["fraction", "numerator", "denominator", "common denominator", "simplify"],
            common_misconceptions=[
                "Adding numerators and denominators separately (1/2 + 1/3 = 2/5)",
                "Not finding common denominator",
                "Forgetting to simplify the final answer",
            ],
            difficulty="easy",
            pre_quiz=[
                QuizQuestion(
                    question="What is a common denominator?",
                    question_type="multiple_choice",
                    options=[
                        "A) The largest denominator",
                        "B) A denominator that both fractions can be converted to",
                        "C) Always 100",
                        "D) The sum of the denominators"
                    ],
                    correct_answer="B",
                    concept="common_denominator"
                ),
                QuizQuestion(
                    question="What is 1/2 + 1/2?",
                    question_type="multiple_choice",
                    options=["A) 1/4", "B) 2/4", "C) 1", "D) 2/2"],
                    correct_answer="C",
                    concept="same_denominator_addition"
                ),
                QuizQuestion(
                    question="Simplify: 4/8",
                    question_type="short_answer",
                    correct_answer="1/2",
                    rubric="Accept 1/2 or 0.5",
                    concept="simplification"
                ),
            ],
            post_quiz=[
                QuizQuestion(
                    question="To add 1/3 + 1/4, what common denominator should you use?",
                    question_type="multiple_choice",
                    options=["A) 7", "B) 12", "C) 3", "D) 4"],
                    correct_answer="B",
                    concept="finding_common_denominator"
                ),
                QuizQuestion(
                    question="What is 1/3 + 1/6?",
                    question_type="multiple_choice",
                    options=["A) 2/9", "B) 2/6", "C) 1/2", "D) 3/6"],
                    correct_answer="C",
                    concept="different_denominator_addition"
                ),
                QuizQuestion(
                    question="Add and simplify: 1/4 + 1/4",
                    question_type="short_answer",
                    correct_answer="1/2",
                    rubric="Accept 1/2 or 0.5. 2/4 gets partial credit if not simplified.",
                    concept="addition_with_simplification"
                ),
            ],
            hints=[
                "Find the least common multiple of the denominators",
                "Convert both fractions to have the same denominator before adding",
                "Always check if your answer can be simplified",
            ],
            examples=[
                "Example: 1/3 + 1/6\nCommon denominator: 6\nConvert: 2/6 + 1/6 = 3/6\nSimplify: 3/6 = 1/2"
            ]
        )


class ProbabilityDomain(TaskDomain):
    """Secondary domain: Basic probability concepts."""

    def get_domain_name(self) -> str:
        return "probability"

    def get_all_tasks(self) -> List[TaskSpec]:
        return [
            self._basic_probability_task(),
            self._conditional_probability_task(),
        ]

    def _basic_probability_task(self) -> TaskSpec:
        """Task on basic probability concepts."""
        return TaskSpec(
            task_id="basic_probability",
            topic="Introduction to Probability",
            description="Learn basic probability: calculating likelihood of events.",
            learning_objectives=[
                "Understand probability as a fraction between 0 and 1",
                "Calculate simple probabilities using favorable/total outcomes",
                "Interpret probability values",
            ],
            key_concepts=["probability", "outcome", "event", "sample space", "favorable outcomes"],
            common_misconceptions=[
                "Thinking probability can be greater than 1",
                "Confusing number of outcomes with probability",
                "Not considering all possible outcomes",
            ],
            difficulty="easy",
            pre_quiz=[
                QuizQuestion(
                    question="What is the range of probability values?",
                    question_type="multiple_choice",
                    options=[
                        "A) 0 to 100",
                        "B) 0 to 1",
                        "C) -1 to 1",
                        "D) Any positive number"
                    ],
                    correct_answer="B",
                    concept="probability_range"
                ),
                QuizQuestion(
                    question="If you flip a fair coin, what is the probability of getting heads?",
                    question_type="multiple_choice",
                    options=["A) 0", "B) 0.5", "C) 1", "D) 2"],
                    correct_answer="B",
                    concept="basic_calculation"
                ),
                QuizQuestion(
                    question="How do you calculate probability?",
                    question_type="short_answer",
                    correct_answer="favorable outcomes divided by total outcomes",
                    rubric="Accept variations mentioning favorable/total or desired/possible",
                    concept="probability_formula"
                ),
            ],
            post_quiz=[
                QuizQuestion(
                    question="If you roll a six-sided die, what is the probability of rolling a 4?",
                    question_type="multiple_choice",
                    options=["A) 1/6", "B) 1/4", "C) 4/6", "D) 1"],
                    correct_answer="A",
                    concept="single_event_probability"
                ),
                QuizQuestion(
                    question="What is the probability of rolling an even number on a six-sided die?",
                    question_type="multiple_choice",
                    options=["A) 1/6", "B) 1/3", "C) 1/2", "D) 2/3"],
                    correct_answer="C",
                    concept="multiple_outcomes"
                ),
                QuizQuestion(
                    question="A bag has 3 red balls and 2 blue balls. What is the probability of drawing a red ball?",
                    question_type="short_answer",
                    correct_answer="3/5",
                    rubric="Accept 3/5, 0.6, or 60%",
                    concept="real_world_application"
                ),
            ],
            hints=[
                "Probability = (favorable outcomes) / (total outcomes)",
                "Make sure you count all possible outcomes",
                "Probability of 0 means impossible, 1 means certain",
            ],
            examples=[
                "Example: Drawing a heart from a deck of cards\nFavorable outcomes: 13 hearts\nTotal outcomes: 52 cards\nProbability: 13/52 = 1/4"
            ]
        )

    def _conditional_probability_task(self) -> TaskSpec:
        """Task on conditional probability."""
        return TaskSpec(
            task_id="conditional_probability",
            topic="Conditional Probability",
            description="Learn about conditional probability: P(A|B) - probability of A given B.",
            learning_objectives=[
                "Understand what conditional probability means",
                "Calculate conditional probabilities",
                "Distinguish between P(A and B) and P(A|B)",
            ],
            key_concepts=["conditional probability", "given", "independent events", "dependent events"],
            common_misconceptions=[
                "Confusing P(A|B) with P(A and B)",
                "Thinking all events are independent",
                "Not updating the sample space for conditional probability",
            ],
            difficulty="medium",
            pre_quiz=[
                QuizQuestion(
                    question="What does P(A|B) mean?",
                    question_type="multiple_choice",
                    options=[
                        "A) Probability of A or B",
                        "B) Probability of A and B",
                        "C) Probability of A given that B occurred",
                        "D) Probability of B given that A occurred"
                    ],
                    correct_answer="C",
                    concept="conditional_notation"
                ),
                QuizQuestion(
                    question="If you draw a card from a deck and don't replace it, are the draws independent?",
                    question_type="multiple_choice",
                    options=["A) Yes", "B) No", "C) Sometimes", "D) Depends on the card"],
                    correct_answer="B",
                    concept="independence"
                ),
                QuizQuestion(
                    question="A bag has 2 red and 3 blue balls. You draw a red ball and don't replace it. What is the probability the next ball is red?",
                    question_type="short_answer",
                    correct_answer="1/4",
                    rubric="Accept 1/4 or 0.25. Only 1 red left out of 4 total.",
                    concept="basic_conditional"
                ),
            ],
            post_quiz=[
                QuizQuestion(
                    question="In conditional probability, what changes compared to regular probability?",
                    question_type="multiple_choice",
                    options=[
                        "A) The formula",
                        "B) The sample space",
                        "C) Nothing",
                        "D) The favorable outcomes only"
                    ],
                    correct_answer="B",
                    concept="conditional_concept"
                ),
                QuizQuestion(
                    question="If P(A and B) = 0.2 and P(B) = 0.5, what is P(A|B)?",
                    question_type="multiple_choice",
                    options=["A) 0.1", "B) 0.3", "C) 0.4", "D) 0.7"],
                    correct_answer="C",
                    concept="conditional_formula"
                ),
                QuizQuestion(
                    question="A class has 10 students: 6 girls and 4 boys. If 4 of the girls and 2 of the boys wear glasses, what is P(wears glasses | girl)?",
                    question_type="short_answer",
                    correct_answer="2/3",
                    rubric="Accept 2/3, 0.667, or ~67%. 4 out of 6 girls wear glasses.",
                    concept="real_world_conditional"
                ),
            ],
            hints=[
                "P(A|B) = P(A and B) / P(B)",
                "When B is given, only consider outcomes where B happened",
                "The sample space shrinks when conditioning on an event",
            ],
            examples=[
                "Example: Two card draws without replacement\nP(2nd card is Ace | 1st card is Ace) = 3/51\nWhy? Only 3 aces left, only 51 cards left"
            ]
        )
