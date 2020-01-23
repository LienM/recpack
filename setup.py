from setuptools import setup, find_packages

setup(
    name="recpack",
    version="0.0.1",
    python_requires="~=3.6",
    packages=find_packages(),
    # tests_require=["pytest"],
    install_requires=[
        "intel-numpy==1.15.1",
        "intel-scipy==1.1.0",
        "google-cloud-storage==1.23.0",
        "pandas==0.25.3",
        "gcsfs==0.6.0"
    ],
    entry_points={
        "console_scripts": [
            "train = synthetic_users_experiments.train:main",
        ]
    },
)
