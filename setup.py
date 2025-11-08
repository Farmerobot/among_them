from setuptools import setup, find_packages

setup(
    name="among_them",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.11,<3.14",
)
