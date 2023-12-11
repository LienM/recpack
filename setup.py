from pathlib import Path
from setuptools import setup, find_packages

SHORT_DESCRIPTION = """Python package for Top-N recommendation based on implicit feedback data."""

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name="recpack",
    version="0.3.6",
    python_requires=">=3.8",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.2, ==1.*",
        "scipy>=1.6.0, ==1.*",
        "scikit-learn>=1.1.1, ==1.*",
        "pandas>=2.1.4, ==2.*",
        "PyYAML>=6.0.1, ==6.*",
        "torch>=2.1.1, ==2.*",
        "tqdm>=4.46.0, ==4.*",
        "hyperopt>=0.2.7, ==0.2.*",
    ],
    extras_require={
        "doc": ["sphinx==4.*", "sphinx-rtd-theme==1.*"],
        "test": ["pytest>=6.2.4, ==6.*", "pytest-cov>=2.12.1, ==2.*"],
    },
    entry_points={},
    description=SHORT_DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://gitlab.com/recpack-maintainers/recpack",
    project_urls={"Documentation": "https://recpack.froomle.ai"},
)
