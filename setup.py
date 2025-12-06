"""
Setup script for TeachTime package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README
readme_file = Path(__file__).parent / "README.md"
if readme_file.exists():
    with open(readme_file, 'r', encoding='utf-8') as f:
        long_description = f.read()
else:
    long_description = "TeachTime: A ReAct Framework for Metric-Guided Language Model Tutors"

setup(
    name="teachtime",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A ReAct framework for metric-guided LLM tutors",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/teach-time",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=[
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "pyyaml>=6.0.1",
        "together>=1.0.0",
        "openai>=1.0.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "scipy>=1.11.0",
        "jupyter>=1.0.0",
        "ipykernel>=6.25.0",
        "pytest>=7.4.0",
        "tqdm>=4.66.0",
        "colorama>=0.4.6",
    ],
    extras_require={
        "dev": [
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "teachtime-run=experiments.run_experiment:main",
            "teachtime-suite=experiments.run_experiment_suite:main",
            "teachtime-pilot=experiments.run_human_pilot:main",
        ],
    },
)
