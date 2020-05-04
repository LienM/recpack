from setuptools import setup, find_packages

setup(
    name="recpack",
    version="0.0.1",
    python_requires="~=3.6",
    packages=find_packages(),
    # tests_require=["pytest"],
    install_requires=[
        "click",
        "numpy",
        "scipy",
        "sklearn",
        "pandas",
        "pytest",
        "pytest-cov",
        "PyYaml",
        "snapy",
        "mmh3",
        "tqdm",
        "dataclasses",
    ],
    entry_points={
        "console_scripts": [
            "run_pipeline = recpack.cli:run_pipeline",
            "run_parameter_generator_pipeline = recpack.cli:run_parameter_generator_pipeline",
        ]
    },
)
