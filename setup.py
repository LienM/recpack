from setuptools import setup, find_packages


setup(
    name="recpack",
    version="0.2.2",
    python_requires=">=3.6",
    packages=find_packages(),
    # tests_require=["pytest"],
    install_requires=[
        "numpy>=1.20.2, ==1.*",
        "scipy>=1.6.0, ==1.*",
        "scikit-learn>=0.24.1, ==0.24.*",
        "pandas>=1.2.1, ==1.*",
        "PyYAML>=5.4.1, ==5.*",
        "torch>=1.9.0, ==1.*",
        "torchtest==0.5",
        "tqdm>=4.46.0, ==4.*",
        "dataclasses==0.6",
    ]
    + ["pytest>=6.2.4, ==6.*", "pytest-cov>=2.12.1, ==2.*"],
    entry_points={"console_scripts": ["run_pipeline = recpack.cli:run_pipeline"]},
)
