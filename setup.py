from setuptools import setup, find_packages


setup(
    name="recpack",
    version="0.1.0",
    python_requires=">=3.6",
    packages=find_packages(),
    # tests_require=["pytest"],
    install_requires=[
        "numpy==1.20.2",
        "scipy==1.6.0",
        "scikit-learn==0.24.1",
        "pandas==1.2.1",
        "PyYAML==5.4.1=",
        "torch==1.7.1",
        "torchtest==0.5",
        "tqdm==4.46.0",
        "dataclasses==0.6",
    ]
    + ["pytest==5.4.1", "pytest-cov==2.8.1"],
    extras_require={"experimental": ["numba==0.50.1"]},
    entry_points={"console_scripts": ["run_pipeline = recpack.cli:run_pipeline"]},
)
