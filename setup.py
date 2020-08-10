from setuptools import setup, find_packages

setup(
    name="recpack",
    version="0.1.0",
    python_requires=">=3.6",
    packages=find_packages(),
    # tests_require=["pytest"],
    install_requires=[
        "click==7.1.1",
        "numpy==1.19.1",
        "scipy==1.5.2",
        "scikit-learn==0.23.2",
        "pandas==1.1.0",
        "PyYAML==5.3.1",
        "snapy==1.0.2",
        "mmh3==2.5.1",
        "torch==1.4.0",
        "torchtest==0.5",
        "tqdm==4.46.0",
        "dataclasses==0.6",
        "joblib==0.14.1",
        "numba==0.50.1"
    ]
    + ["pytest==5.4.1", "pytest-cov==2.8.1"],
    entry_points={
        "console_scripts": [
            "run_pipeline = recpack.cli:run_pipeline"
        ]
    },
)
