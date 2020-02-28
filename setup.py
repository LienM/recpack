from setuptools import setup, find_packages

setup(
    name="recpack",
    version="0.0.1",
    python_requires="~=3.6",
    packages=find_packages(),
    # tests_require=["pytest"],
    install_requires=[
        "numpy",
        "scipy",
        "sklearn",
        "google-cloud-storage==1.23.0",
        "pandas",
        "pytest",
        "pytest-cov",
        "gcsfs==0.6.0"
    ],
    entry_points={},
)
