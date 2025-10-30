from setuptools import setup, find_packages

setup(
    name="translate",
    version="0.2",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'translate': ['data/*.json'],  # Include translation data files
    },
    install_requires=[
        "numpy>=1.20.0",
        "sacrebleu>=2.0.0",
    ],
)
